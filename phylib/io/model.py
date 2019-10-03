# -*- coding: utf-8 -*-

"""Template model."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
from operator import itemgetter
from pathlib import Path
import shutil

import numpy as np
import scipy.io as sio

from .array import _concatenate_virtual_arrays, _index_of, _spikes_in_clusters
from phylib.traces import WaveformLoader
from phylib.utils import Bunch
from phylib.utils._misc import _write_tsv_simple, _read_tsv_simple, read_python
from phylib.utils.geometry import linear_positions

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def read_array(path, mmap_mode=None):
    """Read a binary array in npy or mat format, avoiding nan and inf values."""
    path = Path(path)
    arr_name = path.name
    ext = path.suffix
    if ext == '.mat':  # pragma: no cover
        out = sio.loadmat(path)[arr_name]
    elif ext == '.npy':
        out = np.load(path, mmap_mode=mmap_mode)
    # Filter out nan and inf values.
    # NOTE: do not check for nan/inf values on mmap arrays.
    # TODO: virtual memmap array where the replacement is done on-the-fly when reading the array.
    if mmap_mode is None:
        for w in ('nan', 'inf'):
            errors = getattr(np, 'is' + w)(out)
            if np.any(errors):
                n = np.sum(errors)
                n_tot = errors.size
                logger.warning("%d/%d values are %s in %s, replacing by zero.", n, n_tot, w, path)
                out[errors] = 0
    return out


def write_array(name, arr):
    """Save an array to a binary file."""
    np.save(name, arr)


def from_sparse(data, cols, channel_ids):
    """Convert a sparse structure into a dense one.

    Parameters
    ----------

    data : array-like
        A (n_spikes, n_channels_loc, ...) array with the data.
    cols : array-like
        A (n_spikes, n_channels_loc) array with the channel indices of
        every row in data.
    channel_ids : array-like
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


def load_metadata(filename):
    """Load cluster metadata from a CSV file.

    Return (field_name, dictionary).

    """
    return _read_tsv_simple(filename)


def save_metadata(filename, field_name, metadata):
    """Save metadata in a CSV file."""
    return _write_tsv_simple(filename, field_name, metadata)


#------------------------------------------------------------------------------
# Raw data functions
#------------------------------------------------------------------------------

def _dat_n_samples(filename, dtype=None, n_channels=None, offset=None):
    """Get the number of samples from the size of a dat file."""
    assert dtype is not None
    item_size = np.dtype(dtype).itemsize
    offset = offset if offset else 0
    n_samples = (op.getsize(str(filename)) - offset) // (item_size * n_channels)
    assert n_samples >= 0
    return n_samples


def _dat_to_traces(dat_path, n_channels=None, dtype=None, offset=None):
    """Memmap a dat file."""
    assert dtype is not None
    assert n_channels is not None
    n_samples = _dat_n_samples(dat_path, n_channels=n_channels, dtype=dtype, offset=offset)
    return np.memmap(str(dat_path), dtype=dtype, shape=(n_samples, n_channels), offset=offset)


def load_raw_data(path=None, n_channels_dat=None, dtype=None, offset=None):
    """Load raw data at a given path."""
    if not path:
        return
    path = Path(path)
    if not path.exists():
        logger.warning("Path %s does not exist, trying ephys.raw filename.", path)
        path = path.parent / ('ephys.raw' + path.suffix)
        if not path.exists():
            logger.warning("Error while loading data: File `%s` not found.", path)
            return None
    assert path.exists()
    logger.debug("Loading traces at `%s`.", path)
    dtype = dtype if dtype is not None else np.int16
    return _dat_to_traces(path, n_channels=n_channels_dat, dtype=dtype, offset=offset)


#------------------------------------------------------------------------------
# Channel util functions
#------------------------------------------------------------------------------

def _all_positions_distinct(positions):
    """Return whether all positions are distinct."""
    return len(set(tuple(row) for row in positions)) == len(positions)


def get_closest_channels(channel_positions, channel_index, n=None):
    """Get the channels closest to a given channel on the probe."""
    x = channel_positions[:, 0]
    y = channel_positions[:, 1]
    x0, y0 = channel_positions[channel_index]
    d = (x - x0) ** 2 + (y - y0) ** 2
    out = np.argsort(d)
    if n:
        out = out[:n]
    assert out[0] == channel_index
    return out


#------------------------------------------------------------------------------
# I/O util functions
#------------------------------------------------------------------------------

def _find_first_existing_path(*paths, multiple_ok=True):
    out = []
    for path in paths:
        path = Path(path)
        if path.exists():
            out.append(path)
    if len(out) >= 2 and not multiple_ok:  # pragma: no cover
        raise IOError("Multiple conflicting files exist: %s." % ', '.join((out, path)))
    elif len(out) >= 1:
        return out[0]
    else:
        return None


def _close_memmap(name, obj):
    """Close a memmap array or a list of memmap arrays."""
    if isinstance(obj, np.memmap):
        logger.debug("Close memmap array %s.", name)
        obj._mmap.close()
    elif getattr(obj, 'arrs', None) is not None:
        # Support ConcatenatedArrays.
        _close_memmap('%s.arrs' % name, obj.arrs)
    elif isinstance(obj, (list, tuple)):
        [_close_memmap('%s[]' % name, item) for item in obj]
    elif isinstance(obj, dict):
        [_close_memmap('%s.%s' % (name, n), item) for n, item in obj.items()]


#------------------------------------------------------------------------------
# Template model
#------------------------------------------------------------------------------

class TemplateModel(object):
    """Object holding all data of a KiloSort/phy dataset.

    Constructor
    -----------

    dir_path : str or Path
        Path to the dataset directory
    dat_path : str, Path, or list
        Path to the raw data files.
    dtype : NumPy dtype
        Data type of the raw data file
    offset : int
        Header offset of the binary file
    n_channels_dat : int
        Number of channels in the dat file
    sample_rate : float
        Sampling rate of the data file.
    filter_order : int
        Order of the filter used for waveforms
    hp_filtered : bool
        Whether the raw data file is already high-pass filtered. In that case, disable the
        filtering for the waveform extraction.

    """

    """Number of closest channels used for templates."""
    n_closest_channels = 12

    """Fraction of the peak amplitude required by the closest channels to be kept as best
    channels."""
    amplitude_threshold = 0

    def __init__(self, **kwargs):
        # Default empty values.
        self.dat_path = []
        self.sample_rate = None
        self.n_channels_dat = None

        self.__dict__.update(kwargs)

        # Set dir_path.
        self.dir_path = Path(self.dir_path).resolve()
        assert isinstance(self.dir_path, Path)
        assert self.dir_path.exists()

        # Set dat_path.
        if not self.dat_path:  # pragma: no cover
            self.dat_path = []
        elif not isinstance(self.dat_path, (list, tuple)):
            self.dat_path = [self.dat_path]
        assert isinstance(self.dat_path, (list, tuple))
        self.dat_path = [Path(_).resolve() for _ in self.dat_path]

        self.dtype = getattr(self, 'dtype', np.int16)
        if not self.sample_rate:  # pragma: no cover
            logger.warning("No sample rate was given! Defaulting to 1 Hz.")
        self.sample_rate = float(self.sample_rate or 1.)
        assert self.sample_rate > 0
        self.offset = getattr(self, 'offset', 0)

        self.filter_order = None if getattr(self, 'hp_filtered', False) else 3

        self._load_data()
        self.waveform_loader = self._create_waveform_loader()

    #--------------------------------------------------------------------------
    # Internal loading methods
    #--------------------------------------------------------------------------

    def _load_data(self):
        """Load all data."""
        # Spikes
        self.spike_samples, self.spike_times = self._load_spike_samples()
        ns, = self.n_spikes, = self.spike_times.shape

        # Spike amplitudes.
        self.amplitudes = self._load_amplitudes()
        if self.amplitudes is not None:
            assert self.amplitudes.shape == (ns,)

        # Spike templates.
        self.spike_templates = self._load_spike_templates()
        assert self.spike_templates.shape == (ns,)

        # Spike clusters.
        self.spike_clusters = self._load_spike_clusters()
        assert self.spike_clusters.shape == (ns,)

        # Channels.
        self.channel_mapping = self._load_channel_map()
        self.n_channels = nc = self.channel_mapping.shape[0]
        if self.n_channels_dat:
            assert np.all(self.channel_mapping <= self.n_channels_dat - 1)

        # Channel positions.
        self.channel_positions = self._load_channel_positions()
        assert self.channel_positions.shape == (nc, 2)
        if not _all_positions_distinct(self.channel_positions):  # pragma: no cover
            logger.error(
                "Some channels are on the same position, please check the channel positions file.")
            self.channel_positions = linear_positions(nc)

        # Channel shanks.
        self.channel_shanks = self._load_channel_shanks()
        assert self.channel_shanks.shape == (nc,)

        # Channel probes.
        self.channel_probes = self._load_channel_probes()
        assert self.channel_probes.shape == (nc,)
        self.probes = np.unique(self.channel_probes)
        self.n_probes = len(self.probes)

        # Ordering of the channels in the trace view.
        self.channel_vertical_order = np.argsort(self.channel_positions[:, 1], kind='mergesort')

        # Templates.
        self.sparse_templates = self._load_templates()
        self.n_templates, self.n_samples_waveforms, self.n_channels_loc = \
            self.sparse_templates.data.shape
        if self.sparse_templates.cols is not None:
            assert self.sparse_templates.cols.shape == (self.n_templates, self.n_channels_loc)

        # Whitening.
        try:
            self.wm = self._load_wm()
        except IOError:
            logger.debug("Whitening matrix file not found.")
            self.wm = np.eye(nc)
        assert self.wm.shape == (nc, nc)
        try:
            self.wmi = self._load_wmi()
        except IOError:
            logger.debug("Whitening matrix inverse file not found, computing it.")
            self.wmi = self._compute_wmi(self.wm)
        assert self.wmi.shape == (nc, nc)

        # Similar templates.
        self.similar_templates = self._load_similar_templates()
        assert self.similar_templates.shape == (self.n_templates, self.n_templates)

        # Traces and duration.
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
        self.sparse_features = self._load_features()
        self.features = self.sparse_features.data if self.sparse_features else None
        if self.sparse_features is not None:
            self.n_features_per_channel = self.sparse_features.data.shape[2]

        # Template features.
        self.sparse_template_features = self._load_template_features()
        self.template_features = (
            self.sparse_template_features.data if self.sparse_template_features else None)

        # Spike attributes.
        self.spike_attributes = self._load_spike_attributes()

        # Metadata.
        self.metadata = self._load_metadata()

    def _create_waveform_loader(self):
        # Number of time samples in the templates.
        nsw = self.n_samples_waveforms
        if self.traces is not None:
            return WaveformLoader(
                traces=self.traces,
                spike_samples=self.spike_samples,
                n_samples_waveforms=nsw,
                filter_order=self.filter_order,
                sample_rate=self.sample_rate,
            )

    def _find_path(self, *names, multiple_ok=True):
        """ several """
        full_paths = list(l[0] for l in [list(self.dir_path.glob(name)) for name in names] if l)
        return _find_first_existing_path(*full_paths, multiple_ok=multiple_ok)

    def _read_array(self, path, mmap_mode=None):
        if not path:
            raise IOError(path)
        return read_array(path, mmap_mode=mmap_mode).squeeze()

    def _write_array(self, path, arr):
        return write_array(path, arr)

    def _load_metadata(self):
        """Load cluster metadata from all CSV/TSV files in the data directory."""
        # Files to exclude.
        excluded_names = ('cluster_info',)
        # Get all CSV/TSV files in the directory.
        files = list(self.dir_path.glob('*.csv'))
        files.extend(self.dir_path.glob('*.tsv'))
        metadata = {}
        for filename in files:
            if filename.stem in excluded_names:
                continue
            logger.debug("Load `%s`.", filename)
            try:
                field_name, values = load_metadata(filename)
            except Exception as e:
                logger.debug("Could not load %s: %s.", filename, str(e))
                continue
            metadata[field_name] = values
        return metadata

    #--------------------------------------------------------------------------
    # Specific loading methods
    #--------------------------------------------------------------------------

    def _load_spike_attributes(self):
        """Load all spike_*.npy files, called spike attributes."""
        files = list(self.dir_path.glob('spike_*.npy'))
        spike_attributes = Bunch()
        for filename in files:
            # The part after spike_***
            n = filename.stem[6:]
            # Skip known files.
            if n in ('clusters', 'templates', 'samples', 'times', 'amplitudes'):
                continue
            try:
                arr = self._read_array(filename)
                assert arr.shape[0] == self.n_spikes
                logger.debug("Load %s.", filename)
            except (IOError, AssertionError) as e:
                logger.warning("Unable to open %s: %s.", filename, e)
                continue
            spike_attributes[n] = arr
        return spike_attributes

    def _load_channel_map(self):
        path = self._find_path('channel_map.npy', 'channels._phy_ids*.npy')
        out = self._read_array(path)
        out = np.atleast_1d(out)
        assert out.ndim == 1
        assert out.dtype in (np.uint32, np.int32, np.int64)
        return out

    def _load_channel_positions(self):
        path = self._find_path('channel_positions.npy', 'channels.localCoordinates*.npy')
        out = self._read_array(path)
        out = np.atleast_2d(out)
        assert out.ndim == 2
        return out

    def _load_channel_probes(self):
        try:
            path = self._find_path('channel_probe.npy', 'channels.probes*.npy')
            out = self._read_array(path)
            out = np.atleast_1d(out)
            assert out.ndim == 1
            return out
        except IOError:
            return np.zeros(self.n_channels, dtype=np.int32)

    def _load_channel_shanks(self):
        try:
            path = self._find_path('channel_shanks.npy', 'channels.shanks*.npy')
            out = self._read_array(path).reshape((-1,))
            assert out.ndim == 1
            return out
        except IOError:
            logger.debug("No channel shank file found.")
            return np.zeros(self.n_channels, dtype=np.int32)

    def _load_traces(self, channel_map=None):
        if not self.dat_path:
            return
        paths = self.dat_path
        # Make sure we have a list of paths (virtually-concatenated).
        assert isinstance(paths, (list, tuple))
        n = self.n_channels_dat
        # Memmap all dat files and concatenate them virtually.
        traces = [
            load_raw_data(path, n_channels_dat=n, dtype=self.dtype, offset=self.offset)
            for path in paths]
        traces = [_ for _ in traces if _ is not None]
        traces = _concatenate_virtual_arrays(traces, channel_map)
        return traces

    def _load_amplitudes(self):
        try:
            out = self._read_array(self._find_path('amplitudes.npy', 'spikes.amps*.npy'))
            assert out.ndim == 1
            return out
        except IOError:
            logger.debug("No amplitude file found.")
            return

    def _load_spike_templates(self):
        path = self._find_path('spike_templates.npy', 'spikes.templates*.npy')
        out = self._read_array(path)
        if out.dtype in (np.float32, np.float64):  # pragma: no cover
            out = out.astype(np.int32)
        assert out.dtype in (np.uint32, np.int32, np.int64)
        assert out.ndim == 1
        return out

    def _load_spike_clusters(self):
        path = self._find_path('spike_clusters.npy', 'spikes.clusters*.npy', multiple_ok=False)
        if path is None:
            # Create spike_clusters file if it doesn't exist.
            tmp_path = self._find_path('spike_templates.npy', 'spikes.clusters*.npy')
            path = self.dir_path / 'spike_clusters.npy'
            logger.debug("Copying from %s to %s.", tmp_path, path)
            shutil.copy(tmp_path, path)
        assert path.exists()
        logger.debug("Loading spike clusters.")
        # NOTE: we make a copy in memory so that we can update this array
        # during manual clustering.
        out = self._read_array(path).astype(np.int32)
        assert out.ndim == 1
        return out

    def _load_spike_samples(self):
        # WARNING: "spike_times.npy" is in units of samples. Need to
        # divide by the sampling rate to get spike times in seconds.
        path = self.dir_path / 'spike_times.npy'
        if path.exists():
            # WARNING: spikes_times.npy is in samples !
            samples = self._read_array(path)
            times = samples / self.sample_rate
        else:
            # WARNING: spikes.times.npy is in seconds, not samples !
            times_path = self._find_path('spikes.times*.npy')
            times = self._read_array(times_path)
            samples_path = self._find_path('spikes.samples*.npy')
            if samples_path:
                samples = self._read_array(samples_path)
            else:
                logger.info("Loading spikes.times.npy in seconds, converting to samples.")
                samples = np.round(times * self.sample_rate).astype(np.uint64)
        assert samples.ndim == times.ndim == 1
        return samples, times

    def _load_similar_templates(self):
        try:
            out = self._read_array(self._find_path('similar_templates.npy'))
            out = np.atleast_2d(out)
            assert out.ndim == 2
            return out
        except IOError:
            return np.zeros((self.n_templates, self.n_templates))

    def _load_templates(self):
        logger.debug("Loading templates.")

        # Sparse structure: regular array with col indices.
        try:
            path = self._find_path('templates.npy', 'templates.waveforms*.npy')
            data = self._read_array(path, mmap_mode='r')
            data = np.atleast_3d(data)
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            n_templates, n_samples, n_channels_loc = data.shape
        except IOError:
            return

        try:
            # WARNING: KS2 saves templates_ind.npy (with an s), and note template_ind.npy,
            # so that file is not taken into account here.
            # That means templates.npy is considered as a dense array.
            # Proper fix would be to save templates.npy as a true sparse array, with proper
            # template_ind.npy (without an s).
            path = self._find_path('template_ind.npy')
            cols = self._read_array(path)
            cols = np.atleast_2d(cols)
            assert cols.ndim == 2
            logger.debug("Templates are sparse.")
            assert cols.shape == (n_templates, n_channels_loc)
        except IOError:
            logger.debug("Templates are dense.")
            cols = None

        return Bunch(data=data, cols=cols)

    def _load_wm(self):
        logger.debug("Loading the whitening matrix.")
        out = self._read_array(self._find_path('whitening_mat.npy'))
        out = np.atleast_2d(out)
        assert out.ndim == 2
        return out

    def _load_wmi(self):  # pragma: no cover
        logger.debug("Loading the inverse of the whitening matrix.")
        out = self._read_array(self._find_path('whitening_mat_inv.npy'))
        out = np.atleast_2d(out)
        assert out.ndim == 2
        return out

    def _compute_wmi(self, wm):
        logger.debug("Inversing the whitening matrix %s.", wm.shape)
        try:
            wmi = np.linalg.inv(wm)
        except np.linalg.LinAlgError as e:  # pragma: no cover
            raise ValueError(
                "Error when inverting the whitening matrix: %s.", e)
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
            data = self._read_array(
                self._find_path('pc_features.npy'))
            if data.ndim == 2:  # pragma: no cover
                # Deal with npcs = 1.
                data = data.reshape(data.shape + (1,))
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            data = data.transpose((0, 2, 1))
            n_spikes, n_channels_loc, n_pcs = data.shape
        except IOError:
            return

        try:
            cols = self._read_array(self._find_path('pc_feature_ind.npy'))
            logger.debug("Features are sparse.")
            if cols.ndim == 1:  # pragma: no cover
                # Deal with npcs = 1.
                cols = cols.reshape(cols.shape + (1,))
            assert cols.ndim == 2
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
            data = self._read_array(self._find_path('template_features.npy'), mmap_mode='r')
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

    #--------------------------------------------------------------------------
    # Internal data access methods
    #--------------------------------------------------------------------------

    def _find_best_channels(self, template):
        """Find the best channels for a given template."""
        # Compute the template amplitude on each channel.
        assert template.ndim == 2  # shape: (n_samples, n_channels)
        amplitude = template.max(axis=0) - template.min(axis=0)
        assert amplitude.ndim == 1  # shape: (n_channels,)
        # Find the peak channel.
        best_channel = np.argmax(amplitude)
        max_amp = amplitude[best_channel]
        # Find the channels X% peak.
        peak_channels = np.nonzero(amplitude >= self.amplitude_threshold * max_amp)[0]
        # Find N closest channels.
        close_channels = get_closest_channels(
            self.channel_positions, best_channel, self.n_closest_channels)
        assert best_channel in close_channels
        # Restrict to the channels belonging to the best channel's shank.
        if self.channel_shanks is not None:
            shank = self.channel_shanks[best_channel]  # shank of best channel
            channels_on_shank = np.nonzero(self.channel_shanks == shank)[0]
            close_channels = np.intersect1d(close_channels, channels_on_shank)
        # Keep the intersection.
        channel_ids = np.intersect1d(peak_channels, close_channels)
        # Order the channels by decreasing amplitude.
        order = np.argsort(amplitude[channel_ids])[::-1]
        channel_ids = channel_ids[order]
        amplitude = amplitude[order]
        assert best_channel in channel_ids
        assert amplitude.shape == (len(channel_ids),)
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
        return Bunch(
            template=template,
            amplitude=amplitude,
            best_channel=best_channel,
            channel_ids=channel_ids,
        )

    def _get_template_sparse(self, template_id):
        data, cols = self.sparse_templates.data, self.sparse_templates.cols
        assert cols is not None
        template_w, channel_ids = data[template_id], cols[template_id]

        # Remove unused channels = -1.
        used = channel_ids != -1
        template_w = template_w[:, used]
        channel_ids = channel_ids[used]
        channel_ids = channel_ids.astype(np.uint32)

        # Unwhiten.
        template = self._unwhiten(template_w, channel_ids=channel_ids)
        template = template.astype(np.float32)
        assert template.ndim == 2
        assert template.shape[1] == len(channel_ids)
        # Compute the amplitude and the channel with max amplitude.
        amplitude = template.max(axis=0) - template.min(axis=0)
        best_channel = channel_ids[np.argmax(amplitude)]
        return Bunch(
            template=template,
            amplitude=amplitude,
            best_channel=best_channel,
            channel_ids=channel_ids,
        )

    #--------------------------------------------------------------------------
    # Data access methods
    #--------------------------------------------------------------------------

    def get_template(self, template_id, channel_ids=None):
        """Get data about a template."""
        if self.sparse_templates.cols is not None:
            return self._get_template_sparse(template_id)
        else:
            return self._get_template_dense(template_id, channel_ids=channel_ids)

    def get_waveforms(self, spike_ids, channel_ids=None):
        """Return spike waveforms on specified channels."""
        if self.waveform_loader is None:
            return
        out = self.waveform_loader.get(spike_ids, channel_ids)
        assert out.dtype in (np.float32, np.float64)
        assert out.shape[0] == len(spike_ids)
        if channel_ids is not None:
            assert out.shape[2] == len(channel_ids)
        return out

    def get_features(self, spike_ids, channel_ids):
        """Return sparse features for given spikes."""
        sf = self.sparse_features
        if sf is None:
            return
        _, n_channels_loc, n_pcs = sf.data.shape
        ns = len(spike_ids)
        nc = len(channel_ids)

        # Initialize the output array.
        features = np.empty((ns, n_channels_loc, n_pcs))
        features[:] = np.NAN

        if sf.rows is not None:
            s = np.intersect1d(spike_ids, sf.rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(s, sf.rows)
            # Relative indices of the non-null rows in the output features
            # array.
            rows_out = _index_of(s, spike_ids)
        else:
            rows = spike_ids
            rows_out = slice(None, None, None)
        features[rows_out, ...] = sf.data[rows]

        if sf.cols is not None:
            assert sf.cols.shape[1] == n_channels_loc
            cols = sf.cols[self.spike_templates[spike_ids]]
        else:
            cols = np.tile(np.arange(n_channels_loc), (ns, 1))
        features = from_sparse(features, cols, channel_ids)

        assert features.shape == (ns, nc, n_pcs)
        return features

    def get_template_features(self, spike_ids):
        """Return sparse template features for given spikes."""
        tf = self.sparse_template_features
        if tf is None:
            return
        _, n_templates_loc = tf.data.shape
        ns = len(spike_ids)

        if tf.rows is not None:
            spike_ids = np.intersect1d(spike_ids, tf.rows)
            # Relative indices of the spikes in the self.features_spike_ids
            # array, necessary to load features from all_features which only
            # contains the subset of the spikes.
            rows = _index_of(spike_ids, tf.rows)
        else:
            rows = spike_ids
        template_features = tf.data[rows]

        if tf.cols is not None:
            assert tf.cols.shape[1] == n_templates_loc
            cols = tf.cols[self.spike_templates[spike_ids]]
        else:
            cols = np.tile(np.arange(n_templates_loc), (len(spike_ids), 1))
        template_features = from_sparse(template_features, cols, np.arange(self.n_templates))

        assert template_features.shape[0] == ns
        return template_features

    #--------------------------------------------------------------------------
    # Internal helper methods for public high-level methods
    #--------------------------------------------------------------------------

    def _get_template_from_spikes(self, spike_ids):
        """Get the main template from a set of spikes."""
        # We get the original template ids for the spikes.
        st = self.spike_templates[spike_ids]
        # We find the template with the largest number of spikes from the spike selection.
        template_ids, counts = np.unique(st, return_counts=True)
        ind = np.argmax(counts)
        template_id = template_ids[ind]
        # We load the template.
        template = self.get_template(template_id)
        # We return the template.
        return template

    #--------------------------------------------------------------------------
    # Public high-level methods
    #--------------------------------------------------------------------------

    def describe(self):
        """Display basic information about the dataset."""
        def _print(name, value):
            print("{0: <24}{1}".format(name, value))

        _print('Data files', ', '.join(map(str, self.dat_path)))
        _print('Directory', self.dir_path)
        _print('Duration', '{:.1f}s'.format(self.duration))
        _print('Sample rate', '{:.1f} kHz'.format(self.sample_rate / 1000.))
        _print('Data type', self.dtype)
        _print('# of channels', self.n_channels)
        _print('# of channels (raw)', self.n_channels_dat)
        _print('# of templates', self.n_templates)
        _print('# of spikes', "{:,}".format(self.n_spikes))

    def get_template_spikes(self, template_id):
        """Return the spike ids that belong to a given template."""
        return _spikes_in_clusters(self.spike_templates, [template_id])

    def get_cluster_spikes(self, cluster_id):
        """Return the spike ids that belong to a given template."""
        return _spikes_in_clusters(self.spike_clusters, [cluster_id])

    def get_template_channels(self, template_id):
        """Return the most relevant channels of a template."""
        template = self.get_template(template_id)
        return template.channel_ids

    def get_cluster_channels(self, cluster_id):
        """Return the most relevant channels of a cluster."""
        spike_ids = self.get_cluster_spikes(cluster_id)
        return self._get_template_from_spikes(spike_ids).channel_ids

    def get_template_waveforms(self, template_id):
        """Return the waveforms of a template on the most relevant channels."""
        template = self.get_template(template_id)
        return template.template

    def get_template_spike_waveforms(self, template_id):
        """Return all spike waveforms of a template, on the most relevant channels."""
        spike_ids = self.get_template_spikes(template_id)
        channel_ids = self.get_template_channels(template_id)
        return self.get_waveforms(spike_ids, channel_ids)

    def get_cluster_spike_waveforms(self, cluster_id):
        """Return all spike waveforms of a cluster, on the most relevant channels."""
        spike_ids = self.get_cluster_spikes(cluster_id)
        channel_ids = self.get_cluster_channels(cluster_id)
        return self.get_waveforms(spike_ids, channel_ids)

    @property
    def templates_channels(self):
        """Returns a vector of peak channels for all templates"""
        tmp = self.sparse_templates.data
        n_templates, n_samples, n_channels = tmp.shape
        # Compute the peak channels for each template.
        template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
        assert template_peak_channels.shape == (n_templates,)
        return template_peak_channels

    @property
    def templates_probes(self):
        """Returns a vector of probe index for all templates"""
        return self.channel_probes[self.templates_channels]

    @property
    def templates_amplitudes(self):
        """Returns the average amplitude per cluster"""
        tid = np.unique(self.spike_templates)
        n = np.bincount(self.spike_templates)[tid]
        a = np.bincount(self.spike_templates, weights=self.amplitudes)[tid]
        n[np.isnan(n)] = 1
        return a / n

    @property
    def templates_waveforms_durations(self):
        """Returns a vector of waveform durations (ms) for all templates"""
        tmp = self.sparse_templates.data
        n_templates, n_samples, n_channels = tmp.shape
        # Compute the peak channels for each template.
        template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
        durations = tmp.argmax(axis=1) - tmp.argmin(axis=1)
        ind = np.ravel_multi_index((np.arange(0, n_templates), template_peak_channels),
                                   (n_templates, n_channels), mode='raise', order='C')
        return durations.flatten()[ind].astype(np.float64) / self.sample_rate * 1e3

    #--------------------------------------------------------------------------
    # Saving methods
    #--------------------------------------------------------------------------

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
        out = np.zeros((n_clusters, self.n_samples_waveforms, self.n_channels))
        for i, cluster_id in enumerate(sorted(mean_waveforms)):
            b = mean_waveforms[cluster_id]
            if b.data is not None:
                out[i, :, b.channel_ids] = b.data[0, ...].T
        logger.debug("Save mean waveforms to `%s`.", path)
        np.save(path, out)

    def close(self):
        """Close all memmapped files."""
        for k, v in sorted(self.__dict__.items(), key=itemgetter(0)):
            _close_memmap(k, v)


def _make_abs_path(p, dir_path):
    p = Path(p)
    if not op.isabs(p):
        p = dir_path / p
    if not p.exists():
        logger.warning("File %s does not exist.", p)
    return p


def get_template_params(params_path):
    """Get a dictionary of parameters from a `params.py` file."""
    params_path = Path(params_path)

    params = read_python(params_path)
    params['dtype'] = np.dtype(params['dtype'])

    if 'dir_path' not in params:
        params['dir_path'] = params_path.parent
    params['dir_path'] = Path(params['dir_path'])
    assert params['dir_path'].is_dir()
    assert params['dir_path'].exists()

    if isinstance(params['dat_path'], str):
        params['dat_path'] = [params['dat_path']]
    params['dat_path'] = [_make_abs_path(_, params['dir_path']) for _ in params['dat_path']]
    return params


def load_model(params_path):
    """Return a TemplateModel instance from a path to a `params.py` file."""
    return TemplateModel(**get_template_params(params_path))
