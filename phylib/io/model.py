# -*- coding: utf-8 -*-

"""Template model."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
from pathlib import Path
import shutil

import numpy as np
import scipy.io as sio

from .array import (
    _concatenate_virtual_arrays,
    _index_of,
    _spikes_in_clusters,
)
from phylib.traces import WaveformLoader
from phylib.utils import Bunch
from phylib.utils._misc import _write_tsv, _read_tsv


logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def read_array(path):
    path = Path(path)
    arr_name = path.name
    ext = path.suffix
    if ext == '.mat':  # pragma: no cover
        return sio.loadmat(path)[arr_name]
    elif ext == '.npy':
        return np.load(path, mmap_mode='r')


def write_array(name, arr):
    np.save(name, arr)


def load_metadata(filename):
    """Load cluster metadata from a CSV file.

    Return (field_name, dictionary).

    """
    return _read_tsv(filename)


def save_metadata(filename, field_name, metadata):
    """Save metadata in a CSV file."""
    return _write_tsv(filename, field_name, metadata)


def _dat_n_samples(filename, dtype=None, n_channels=None, offset=None):
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(filename) - offset) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def _dat_to_traces(dat_path, n_channels=None, dtype=None, offset=None):
    assert dtype is not None
    assert n_channels is not None
    n_samples = _dat_n_samples(dat_path, n_channels=n_channels, dtype=dtype, offset=offset)
    return np.memmap(dat_path, dtype=dtype, shape=(n_samples, n_channels), offset=offset)


def load_raw_data(path=None, n_channels_dat=None, dtype=None, offset=None):
    if not path:
        return
    path = Path(path)
    if not path.exists():
        path = path.parent / ('ephys.raw' + path.suffix)
        if not path.exists():
            logger.warning(
                "Error while loading data: File `%s` not found.", path)
            return None
    assert path.exists()
    logger.debug("Loading traces at `%s`.", path)
    dtype = dtype if dtype is not None else np.int16
    return _dat_to_traces(path, n_channels=n_channels_dat, dtype=dtype, offset=offset)


def get_closest_channels(channel_positions, channel_index, n=None):
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    return out


def from_sparse(data, cols, channel_ids):
    """Convert a sparse structure into a dense one.

    Arguments:

    data : array
        A (n_spikes, n_channels_loc, ...) array with the data.
    cols : array
        A (n_spikes, n_channels_loc) array with the channel indices of
        every row in data.
    channel_ids : array
        List of requested channel ids (columns).

    """
    # The axis in the data that contains the channels.
    if len(channel_ids) != len(np.unique(channel_ids)):
        raise NotImplementedError("Multiple identical requested channels "
                                  "in from_sparse().")
    channel_axis = 1
    shape = list(data.shape)
    assert data.ndim >= 2
    assert cols.ndim == 2
    assert data.shape[:2] == cols.shape
    n_spikes, n_channels_loc = shape[:2]
    # NOTE: we ensure here that `col` contains integers.
    c = cols.flatten().astype(np.int32)
    # Remove columns that do not belong to the specified channels.
    c[~np.in1d(c, channel_ids)] = -1
    assert np.all(np.in1d(c, np.r_[channel_ids, -1]))
    # Convert column indices to relative indices given the specified
    # channel_ids.
    cols_loc = _index_of(c, np.r_[channel_ids, -1]).reshape(cols.shape)
    assert cols_loc.shape == (n_spikes, n_channels_loc)
    n_channels = len(channel_ids)
    # Shape of the output array.
    out_shape = shape
    # The channel dimension contains the number of requested channels.
    # The last column contains irrelevant values.
    out_shape[channel_axis] = n_channels + 1
    out = np.zeros(out_shape, dtype=data.dtype)
    x = np.tile(np.arange(n_spikes)[:, np.newaxis],
                (1, n_channels_loc))
    assert x.shape == cols_loc.shape == data.shape[:2]
    out[x, cols_loc, ...] = data
    # Remove the last column with values outside the specified
    # channels.
    out = out[:, :-1, ...]
    return out


def _find_first_existing_path(*paths, multiple_ok=True):
    out = None
    for path in paths:
        path = Path(path)
        if path.exists():
            if out is not None and not multiple_ok:  # pragma: no cover
                raise IOError("Multiple conflicting files exist: %s." % ', '.join((out, path)))
            out = path
        if multiple_ok:
            break
    return out


#------------------------------------------------------------------------------
# Template model
#------------------------------------------------------------------------------

class TemplateModel(object):
    n_closest_channels = 16
    amplitude_threshold = .25

    def __init__(self, dat_path=None, **kwargs):
        dat_path = dat_path or ''
        dat_path = Path(dat_path).expanduser().resolve()
        dir_path = dat_path.parent if dat_path else os.getcwd()
        self.dat_path = dat_path
        self.dir_path = dir_path
        self.__dict__.update(kwargs)

        self.dtype = getattr(self, 'dtype', np.int16)
        self.sample_rate = float(self.sample_rate)
        assert self.sample_rate > 0
        self.offset = getattr(self, 'offset', 0)

        self.filter_order = None if getattr(self, 'hp_filtered', False) else 3

        self._load_data()
        self.waveform_loader = self._create_waveform_loader()

    def describe(self):
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))

        _print('Data file', self.dat_path)
        _print('Data shape',
               'None' if self.traces is None else str(self.traces.shape))
        _print('Number of channels', self.n_channels)
        _print('Duration', '{:.1f}s'.format(self.duration))
        _print('Number of spikes', self.n_spikes)
        _print('Number of templates', self.n_templates)
        _print('Features shape',
               'None' if self.features is None else str(self.features.shape))

    def spikes_in_template(self, template_id):
        return _spikes_in_clusters(self.spike_templates, [template_id])

    def _load_data(self):
        sr = self.sample_rate

        # Spikes.
        self.spike_samples = self._load_spike_samples()
        self.spike_times = self.spike_samples / sr
        ns, = self.n_spikes, = self.spike_times.shape

        self.amplitudes = self._load_amplitudes()
        assert self.amplitudes.shape == (ns,)

        self.spike_templates = self._load_spike_templates()
        assert self.spike_templates.shape == (ns,)

        self.spike_clusters = self._load_spike_clusters()
        assert self.spike_clusters.shape == (ns,)

        # Channels.
        self.channel_mapping = self._load_channel_map()
        self.n_channels = nc = self.channel_mapping.shape[0]
        assert np.all(self.channel_mapping <= self.n_channels_dat - 1)

        self.channel_positions = self._load_channel_positions()
        assert self.channel_positions.shape == (nc, 2)

        self.channel_vertical_order = np.argsort(self.channel_positions[:, 1],
                                                 kind='mergesort')

        # Templates.
        self.sparse_templates = self._load_templates()
        self.n_templates = self.sparse_templates.data.shape[0]
        self.n_samples_templates = self.sparse_templates.data.shape[1]
        self.n_channels_loc = self.sparse_templates.data.shape[2]
        if self.sparse_templates.cols is not None:
            assert self.sparse_templates.cols.shape == (self.n_templates,
                                                        self.n_channels_loc)

        # Whitening.
        try:
            self.wm = self._load_wm()
        except IOError:
            logger.warning("Whitening matrix is not available.")
            self.wm = np.eye(nc)
        assert self.wm.shape == (nc, nc)
        try:
            self.wmi = self._load_wmi()
        except IOError:
            self.wmi = self._compute_wmi(self.wm)
        assert self.wmi.shape == (nc, nc)

        self.similar_templates = self._load_similar_templates()
        assert self.similar_templates.shape == (self.n_templates, self.n_templates)

        self.traces = self._load_traces(self.channel_mapping)
        if self.traces is not None:
            self.duration = self.traces.shape[0] / float(self.sample_rate)
        else:
            self.duration = self.spike_times[-1]
        if self.spike_times[-1] > self.duration:  # pragma: no cover
            logger.debug(
                "There are %d/%d spikes after the end of the recording.",
                np.sum(self.spike_times > self.duration), self.n_spikes)

        # Features.
        f = self._load_features()
        if f is not None:
            self.features = f.data
            self.n_features_per_channel = self.features.shape[2]
            self.features_cols = f.cols
            self.features_rows = f.rows
        else:
            self.features = None
            self.features_cols = None
            self.features_rows = None

        tf = self._load_template_features()
        if tf is not None:
            self.template_features = tf.data
            self.template_features_cols = tf.cols
            self.template_features_rows = tf.rows
        else:
            self.template_features = None

        self.metadata = self._load_metadata()

    def _create_waveform_loader(self):
        # Number of time samples in the templates.
        nsw = self.n_samples_templates
        if self.traces is not None:
            return WaveformLoader(traces=self.traces,
                                  spike_samples=self.spike_samples,
                                  n_samples_waveforms=nsw,
                                  filter_order=self.filter_order,
                                  sample_rate=self.sample_rate,
                                  )

    def _find_path(self, *names, multiple_ok=True):
        return _find_first_existing_path(
            *(self.dir_path / name for name in names), multiple_ok=multiple_ok)

    def _read_array(self, path):
        if not path:
            raise IOError(path)
        return read_array(path).squeeze()

    def _write_array(self, path, arr):
        return write_array(path, arr)

    def _load_metadata(self):
        """Load cluster metadata from all CSV/TSV files in the data directory."""
        files = list(self.dir_path.glob('*.csv'))
        files.extend(self.dir_path.glob('*.tsv'))
        metadata = {}
        for filename in files:
            logger.debug("Load `%s`.", filename)
            try:
                field_name, values = load_metadata(filename)
            except Exception as e:
                logger.warning("Could not load %s: %s.", filename, str(e))
                continue
            metadata[field_name] = values
        return metadata

    @property
    def metadata_fields(self):
        """List of metadata fields."""
        return sorted(self.metadata)

    def get_metadata(self, name):
        """Return a dictionary {cluster_id: value} for a cluster metadata
        field."""
        return self.metadata.get(name, {})

    def save_metadata(self, name, values):
        """Save a dictionary {cluster_id: value} with cluster metadata in
        a TSV file."""
        path = self.dir_path / ('cluster_%s.tsv' % name)
        logger.debug("Save cluster metadata to `%s`.", path)
        # Remove empty values.
        save_metadata(
            path, name, {c: v for c, v in values.items() if v is not None})

    def save_spike_clusters(self, spike_clusters):
        """Save the spike clusters."""
        path = self._find_path('spike_clusters.npy', 'spikes.clusters.npy', multiple_ok=False)
        logger.debug("Save spike clusters to `%s`.", path)
        np.save(path, spike_clusters)

    def save_mean_waveforms(self, mean_waveforms):
        """Save the mean waveforms as a single array."""
        path = self.dir_path / 'clusters.meanWaveforms.npy'
        n_clusters = len(mean_waveforms)
        out = np.zeros((n_clusters, self.n_samples_templates, self.n_channels))
        for i, cluster_id in enumerate(sorted(mean_waveforms)):
            b = mean_waveforms[cluster_id]
            if b.data is not None:
                out[i, :, b.channel_ids] = b.data[0, ...].T
        logger.debug("Save mean waveforms to `%s`.", path)
        np.save(path, out)

    def _load_channel_map(self):
        path = self._find_path('channel_map.npy', 'channels.rawRow.npy')
        out = self._read_array(path)
        assert out.dtype in (np.uint32, np.int32, np.int64)
        return out

    def _load_channel_positions(self):
        path = self._find_path('channel_positions.npy', 'channels.sitePositions.npy')
        return self._read_array(path)

    def _load_traces(self, channel_map=None):
        if not self.dat_path:
            return
        paths = self.dat_path
        # Make sure we have a list of paths (virtually-concatenated).
        if not isinstance(paths, list):
            paths = [paths]
        n = self.n_channels_dat
        # Memmap all dat files and concatenate them virtually.
        traces = [
            load_raw_data(path, n_channels_dat=n, dtype=self.dtype, offset=self.offset)
            for path in paths]
        traces = [_ for _ in traces if _ is not None]
        scaling = 1. / 255 if self.dtype == np.int16 else None
        traces = _concatenate_virtual_arrays(traces, channel_map, scaling=scaling)
        return traces

    def _load_amplitudes(self):
        return self._read_array(self._find_path('amplitudes.npy', 'spikes.amps.npy'))

    def _load_spike_templates(self):
        path = self._find_path('spike_templates.npy', 'ks2/spikes.clusters.npy')
        out = self._read_array(path)
        if out.dtype in (np.float32, np.float64):  # pragma: no cover
            out = out.astype(np.int32)
        assert out.dtype in (np.uint32, np.int32, np.int64)
        return out

    def _load_spike_clusters(self):
        path = self._find_path('spike_clusters.npy', 'spikes.clusters.npy', multiple_ok=False)
        if path is None:
            # Create spike_clusters file if it doesn't exist.
            tmp_path = self._find_path('spike_templates.npy', 'ks2/spikes.clusters.npy')
            path = self.dir_path / 'spike_clusters.npy'
            logger.debug("Copying from %s to %s.", tmp_path, path)
            shutil.copy(tmp_path, path)
        assert path.exists()
        logger.debug("Loading spike clusters.")
        # NOTE: we make a copy in memory so that we can update this array
        # during manual clustering.
        out = self._read_array(path).astype(np.int32)
        return out

    def _load_spike_samples(self):
        # WARNING: "spike_times.npy" is in units of samples. Need to
        # divide by the sampling rate to get spike times in seconds.
        path = self.dir_path / 'spike_times.npy'
        if path.exists():
            return self._read_array(path)
        else:
            # WARNING: spikes.times.npy is in seconds, not samples !
            path = self.dir_path / 'spikes.times.npy'
            logger.info("Loading spikes.times.npy in seconds, converting to samples.")
            spike_times = self._read_array(path)
            return (spike_times * self.sample_rate).astype(np.uint64)

    def _load_similar_templates(self):
        return self._read_array(self._find_path('similar_templates.npy'))

    def _load_templates(self):
        logger.debug("Loading templates.")

        # Sparse structure: regular array with col indices.
        try:
            path = self._find_path('templates.npy', 'clusters.templateWaveforms.npy')
            data = self._read_array(path)
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            n_templates, n_samples, n_channels_loc = data.shape
        except IOError:
            return

        try:
            cols = self._read_array(self._find_path('template_ind.npy'))
            logger.debug("Templates are sparse.")
            assert cols.shape == (n_templates, n_channels_loc)
        except IOError:
            logger.debug("Templates are dense.")
            cols = None

        return Bunch(data=data, cols=cols)

    def _load_wm(self):
        logger.debug("Loading the whitening matrix.")
        return self._read_array(self._find_path('whitening_mat.npy'))

    def _load_wmi(self):
        logger.debug("Loading the inverse of the whitening matrix.")
        return self._read_array(self._find_path('whitening_mat_inv.npy'))

    def _compute_wmi(self, wm):
        logger.debug("Inversing the whitening matrix %s.", wm.shape)
        wmi = np.linalg.inv(wm)
        self._write_array(self.dir_path / 'whitening_mat_inv.npy', wmi)
        return wmi

    def _unwhiten(self, x, channel_ids=None):
        mat = self.wmi
        if channel_ids is not None:
            mat = mat[np.ix_(channel_ids, channel_ids)]
            assert mat.shape == (len(channel_ids),) * 2
        assert x.shape[1] == mat.shape[0]
        return np.dot(np.ascontiguousarray(x),
                      np.ascontiguousarray(mat))

    def _load_features(self):

        # Sparse structure: regular array with row and col indices.
        try:
            logger.debug("Loading features.")
            data = self._read_array(self._find_path('pc_features.npy')).transpose((0, 2, 1))
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            n_spikes, n_channels_loc, n_pcs = data.shape
        except IOError:
            return

        try:
            cols = self._read_array(self._find_path('pc_feature_ind.npy'))
            logger.debug("Features are sparse.")
            assert cols.shape == (self.n_templates, n_channels_loc)
        except IOError:
            logger.debug("Features are dense.")
            cols = None

        try:
            rows = self._read_array(self._find_path('pc_feature_spike_ids.npy'))
            assert rows.shape == (n_spikes,)
        except IOError:
            rows = None

        return Bunch(data=data, cols=cols, rows=rows)

    def _load_template_features(self):

        # Sparse structure: regular array with row and col indices.
        try:
            logger.debug("Loading template features.")
            data = self._read_array(self._find_path('template_features.npy'))
            assert data.dtype in (np.float32, np.float64)
            assert data.ndim == 2
            n_spikes, n_channels_loc = data.shape
        except IOError:
            return

        try:
            cols = self._read_array(self._find_path('template_feature_ind.npy'))
            logger.debug("Template features are sparse.")
            assert cols.shape == (self.n_templates, n_channels_loc)
        except IOError:
            cols = None
            logger.debug("Template features are dense.")

        try:
            rows = self._read_array(self._find_path('template_feature_spike_ids.npy'))
            assert rows.shape == (n_spikes,)
        except IOError:
            rows = None

        return Bunch(data=data, cols=cols, rows=rows)

    def _get_template_sparse(self, template_id):
        data, cols = self.sparse_templates.data, self.sparse_templates.cols
        assert cols is not None
        template_w, channel_ids = data[template_id], cols[template_id]
        # Remove unused channels = -1.
        used = channel_ids != -1
        template_w = template_w[:, used]
        channel_ids = channel_ids[used]
        # Unwhiten.
        template = self._unwhiten(template_w, channel_ids=channel_ids)
        template = template.astype(np.float32)
        assert template.ndim == 2
        assert template.shape[1] == len(channel_ids)
        # Compute the amplitude and the channel with max amplitude.
        amplitude = template.max(axis=0) - template.min(axis=0)
        best_channel = np.argmax(amplitude)
        b = Bunch(template=template,
                  amplitude=amplitude,
                  best_channel=best_channel,
                  channel_ids=channel_ids,
                  )
        return b

    def _find_best_channels(self, template):
        # Compute the template amplitude on each channel.
        amplitude = template.max(axis=0) - template.min(axis=0)
        # Find the peak channel.
        best_channel = np.argmax(amplitude)
        max_amp = amplitude[best_channel]
        # Find the channels X% peak.
        peak_channels = np.nonzero(amplitude > self.amplitude_threshold * max_amp)[0]
        # Find N closest channels.
        close_channels = get_closest_channels(
            self.channel_positions, best_channel, self.n_closest_channels)
        # Keep the intersection.
        channel_ids = np.intersect1d(peak_channels, close_channels)
        return channel_ids, amplitude, best_channel

    def _get_template_dense(self, template_id, channel_ids=None):
        """Return data for one template."""
        template_w = self.sparse_templates.data[template_id, ...]
        template = self._unwhiten(template_w).astype(np.float32)
        assert template.ndim == 2
        channel_ids_, amplitude, best_channel = self._find_best_channels(template)
        channel_ids = channel_ids if channel_ids is not None else channel_ids_
        template = template[:, channel_ids]
        assert template.ndim == 2
        assert template.shape[1] == channel_ids.shape[0]
        b = Bunch(template=template,
                  amplitude=amplitude,
                  best_channel=best_channel,
                  channel_ids=channel_ids,
                  )
        return b

    def get_template(self, template_id, channel_ids=None):
        if self.sparse_templates.cols is not None:
            return self._get_template_sparse(template_id)
        else:
            return self._get_template_dense(template_id, channel_ids=channel_ids)

    def get_waveforms(self, spike_ids, channel_ids):
        """Return several waveforms on specified channels."""
        if self.waveform_loader is None:
            return
        out = self.waveform_loader.get(spike_ids, channel_ids)
        assert out.dtype in (np.float32, np.float64)
        assert out.shape[0] == len(spike_ids)
        assert out.shape[2] == len(channel_ids)
        return out

    def get_features(self, spike_ids, channel_ids):
        """Return sparse features for given spikes."""
        data = self.features
        if data is None:
            return
        _, n_channels_loc, n_pcs = data.shape
        ns = len(spike_ids)
        nc = len(channel_ids)

        # Initialize the output array.
        features = np.empty((ns, n_channels_loc, n_pcs))
        features[:] = np.NAN

        if self.features_rows is not None:
            s = np.intersect1d(spike_ids, self.features_rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(s, self.features_rows)
            # Relative indices of the non-null rows in the output features
            # array.
            rows_out = _index_of(s, spike_ids)
        else:
            rows = spike_ids
            rows_out = slice(None, None, None)
        features[rows_out, ...] = data[rows]

        if self.features_cols is not None:
            assert self.features_cols.shape[1] == n_channels_loc
            cols = self.features_cols[self.spike_templates[spike_ids]]
        else:
            cols = np.tile(np.arange(n_channels_loc), (ns, 1))
        features = from_sparse(features, cols, channel_ids)

        assert features.shape == (ns, nc, n_pcs)
        return features

    def get_template_features(self, spike_ids):
        """Return sparse template features for given spikes."""
        data = self.template_features
        if data is None:
            return
        _, n_templates_loc = data.shape
        ns = len(spike_ids)

        if self.template_features_rows is not None:
            spike_ids = np.intersect1d(spike_ids, self.features_rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(spike_ids, self.template_features_rows)
        else:
            rows = spike_ids
        template_features = data[rows]

        if self.template_features_cols is not None:
            assert self.template_features_cols.shape[1] == n_templates_loc
            cols = self.template_features_cols[self.spike_templates[spike_ids]]
        else:
            cols = np.tile(np.arange(n_templates_loc), (len(spike_ids), 1))
        template_features = from_sparse(template_features, cols, np.arange(self.n_templates))

        assert template_features.shape[0] == ns
        return template_features
