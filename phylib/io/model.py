# -*- coding: utf-8 -*-

"""Template model."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
from operator import itemgetter
from pathlib import Path
import shutil

import numpy as np
# from numpy.lib.format import open_memmap
import scipy.io as sio
# from tqdm import tqdm

from .array import _index_of, _spikes_in_clusters, _spikes_per_cluster, SpikeSelector
from .traces import (
    get_ephys_reader, RandomEphysReader, extract_waveforms,
    get_spike_waveforms, export_waveforms)
from phylib.utils import Bunch
from phylib.utils._misc import _write_tsv_simple, read_tsv, read_python
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
    """Load cluster metadata from a TSV file.

    Return {field_name: dictionary cluster_id => value}.

    """
    data = read_tsv(filename)
    if not data:  # pragma: no cover
        return {}
    out = {}
    for d in data:
        if 'cluster_id' in d:
            cluster_id = d['cluster_id']
            for field, value in d.items():
                if field != 'cluster_id':
                    if field not in out:
                        out[field] = {}
                    out[field][cluster_id] = value
    return out


def save_metadata(filename, field_name, metadata):
    """Save metadata in a CSV file."""
    return _write_tsv_simple(filename, field_name, metadata)


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
# PC computation
#------------------------------------------------------------------------------

def _compute_pcs(x, npcs):
    """Compute the PCs of an array x, where each row is an observation.
    x can be a 2D or 3D array. In the latter case, the PCs are computed
    and concatenated iteratively along the last axis."""

    # Ensure x is a 3D array.
    assert x.ndim == 3
    # Ensure double precision
    x = x.astype(np.float64)

    nspikes, nsamples, nchannels = x.shape

    cov_reg = np.eye(nsamples)
    assert cov_reg.shape == (nsamples, nsamples)

    pcs_list = []
    # Loop over channels
    for channel in range(nchannels):
        x_channel = x[:, :, channel]
        # Compute cov matrix for the channel
        assert x_channel.ndim == 2
        # Don't compute the cov matrix if there are no unmasked spikes
        # on that channel.
        alpha = 1. / nspikes
        if x_channel.shape[0] <= 1:
            cov = alpha * cov_reg
        else:
            cov_channel = np.cov(x_channel, rowvar=0)
            assert cov_channel.shape == (nsamples, nsamples)
            cov = alpha * cov_reg + cov_channel
        # Compute the eigenelements
        vals, vecs = np.linalg.eigh(cov)
        pcs = vecs.T.astype(np.float32)[np.argsort(vals)[::-1]]
        # Take the first npcs components.
        pcs_list.append(pcs[:npcs, ...])
    # Return the concatenation of the PCs on all channels, along the 3d axis,
    # except if there is only one element in the 3d axis. In this case
    # we convert to a 2D array.
    pcs = np.dstack(pcs_list)
    assert pcs.ndim == 3
    return pcs


def _project_pcs(x, pcs):
    """Project data points onto principal components.
    Arguments:
      * x: a 2D array.
      * pcs: the PCs as returned by `compute_pcs`.
    """
    assert x.ndim == 3
    assert pcs.ndim == 3
    features = []
    # for i, waveform in enumerate(x):
    #     assert waveform.ndim == 2
    #     # pcs.shape is (3, n_samples, n_channels)
    #     # waveform.shape is (n_samples, n_channels)
    #     features.append(np.einsum('ijk,jk->ki', pcs, waveform))
    features = np.einsum('ijk,ljk->lki', pcs, x)
    # features = np.stack(features, axis=0)
    assert features.ndim == 3
    return features


def compute_features(waveforms):
    assert waveforms.ndim == 3
    nspk, nsmp, nc = waveforms.shape
    pcs = _compute_pcs(waveforms, 3)
    assert pcs.ndim == 3
    features = _project_pcs(waveforms, pcs)
    assert features.ndim == 3
    assert features.shape == (nspk, nc, 3)
    return features


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
    elif getattr(obj, 'arrs', None) is not None:  # pragma: no cover
        # Support ConcatenatedArrays.
        # NOTE: no longer used since EphysTraces
        _close_memmap('%s.arrs' % name, obj.arrs)
    elif isinstance(obj, (list, tuple)):
        [_close_memmap('%s[]' % name, item) for item in obj]
    elif isinstance(obj, dict):
        [_close_memmap('%s.%s' % (name, n), item) for n, item in obj.items()]


#------------------------------------------------------------------------------
# Template model
#------------------------------------------------------------------------------

# Special spike_*.npy files that should not be considered as "spike attributes".
SKIP_SPIKE_ATTRS = ('clusters', 'templates', 'samples', 'times', 'times_reordered', 'amplitudes')


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

        self._load_data()

    #--------------------------------------------------------------------------
    # Internal loading methods
    #--------------------------------------------------------------------------

    def _load_data(self):
        """Load all data."""
        # Spikes
        self.spike_samples, self.spike_times = self._load_spike_samples()
        ns, = self.n_spikes, = self.spike_times.shape

        # Make sure the spike times are increasing.
        if not np.all(np.diff(self.spike_times) >= 0):
            raise ValueError("The spike times must be increasing.")

        # Spike amplitudes.
        self.amplitudes = self._load_amplitudes()
        if self.amplitudes is not None:
            assert self.amplitudes.shape == (ns,)

        # Spike templates.
        self.spike_templates = self._load_spike_templates()
        assert self.spike_templates.shape == (ns,)

        # Unique template ids.
        self.template_ids = np.unique(self.spike_templates)

        # Spike clusters.
        self.spike_clusters = self._load_spike_clusters()
        assert self.spike_clusters.shape == (ns,)

        # Unique cluster ids.
        self.cluster_ids = np.unique(self.spike_clusters)

        # Spike reordering.
        self.spike_times_reordered = self._load_spike_reorder()
        if self.spike_times_reordered is not None:
            assert self.spike_times_reordered.shape == (ns,)

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

        # Templates.
        self.sparse_templates = self._load_templates()
        if self.sparse_templates is not None:
            self.n_templates, self.n_samples_waveforms, self.n_channels_loc = \
                self.sparse_templates.data.shape
            if self.sparse_templates.cols is not None:
                assert self.sparse_templates.cols.shape == (self.n_templates, self.n_channels_loc)
        else:  # pragma: no cover
            self.n_templates = self.spike_templates.max() + 1
            self.n_samples_waveforms = 0
            self.n_channels_loc = 0

        # Clusters waveforms
        if not np.all(self.spike_clusters == self.spike_templates) and \
                self.sparse_templates.cols is None:
            self.merge_map, self.nan_idx = self.get_merge_map()
            self.sparse_clusters = self.cluster_waveforms()
            self.n_clusters = self.spike_clusters.max() + 1
        else:
            self.merge_map = {}
            self.nan_idx = []
            self.sparse_clusters = self.sparse_templates
            self.n_clusters = self.spike_templates.max() + 1

        # Spike waveforms (optional, otherwise fetched from raw data as needed).
        self.spike_waveforms = self._load_spike_waveforms()

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
            self.duration = self.traces.duration
        else:
            self.duration = self.spike_times[-1]
        if self.spike_times[-1] > self.duration:  # pragma: no cover
            logger.warning(
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

    def _find_path(self, *names, multiple_ok=True, mandatory=True):
        full_paths = list(l[0] for l in [list(self.dir_path.glob(name)) for name in names] if l)
        path = _find_first_existing_path(*full_paths, multiple_ok=multiple_ok)
        if mandatory and not path:
            raise IOError(
                "None of these files could be found in %s: %s." %
                (self.dir_path, ', '.join(names)))
        return path

    def _read_array(self, path, mmap_mode=None):
        if not path:
            raise IOError()
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
            logger.debug("Load `%s`.", filename.name)
            try:
                for field, data in load_metadata(filename).items():
                    metadata[field] = data
            except Exception as e:
                logger.warning("Error when reading %s: %s.", filename.name, str(e))
                continue
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
            if n in SKIP_SPIKE_ATTRS:
                continue
            try:
                arr = self._read_array(filename)
                assert arr.shape[0] == self.n_spikes
                logger.debug("Load %s.", filename.name)
            except (IOError, AssertionError) as e:
                logger.warning("Unable to open %s: %s.", filename.name, e)
                continue
            spike_attributes[n] = arr
        return spike_attributes

    def _load_channel_map(self):
        path = self._find_path('channel_map.npy', 'channels.rawInd*.npy')
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
            if os.environ.get('PHY_VIRTUAL_RAW_DATA', None):  # pragma: no cover
                n_samples = int((self.spike_times[-1] + 1) * self.sample_rate)
                return RandomEphysReader(n_samples, len(channel_map), sample_rate=self.sample_rate)
            return
        n = self.n_channels_dat
        # self.dat_path could be any object accepted by get_ephys_reader().
        traces = get_ephys_reader(
            self.dat_path, n_channels_dat=n, dtype=self.dtype, offset=self.offset,
            sample_rate=self.sample_rate)
        if traces is not None:
            traces = traces[:, channel_map]  # lazy permutation on the channel axis
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
        uc = np.unique(out)
        if np.max(uc) - np.min(uc) + 1 != uc.size:
            logger.warning(
                "Unreferenced clusters found in templates (generally not a problem)")
        assert out.dtype in (np.uint32, np.int32, np.int64)
        assert out.ndim == 1
        return out

    def _load_spike_clusters(self):
        path = self._find_path(
            'spike_clusters.npy', 'spikes.clusters*.npy', multiple_ok=False, mandatory=False)
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
        uc = np.unique(out)
        if np.max(uc) - np.min(uc) + 1 != uc.size:
            logger.warning(
                "Unreferenced clusters found in spike_clusters (generally not a problem)")
        assert out.ndim == 1
        return out

    def _load_spike_reorder(self):
        """Load spike_times_reordered.npy, a 1D array with alternative spike times, in number of
        samples (not in seconds)."""
        path = self.dir_path / 'spike_times_reordered.npy'
        if path.exists():
            logger.debug("Loading spike times reordered.")
            samples = self._read_array(path).squeeze()
            times = samples / self.sample_rate
            assert times.shape == (self.n_spikes,)
            return times

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
            times_path = self._find_path('spikes.times*.npy', mandatory=False)
            times = self._read_array(times_path)
            samples_path = self._find_path('spikes.samples*.npy', mandatory=False)
            if samples_path:
                samples = self._read_array(samples_path)
            else:
                logger.info("Loading spikes.times.npy in seconds, converting to samples.")
                samples = np.round(times * self.sample_rate).astype(np.uint64)
        assert samples.ndim == times.ndim == 1
        return samples, times

    def _load_spike_waveforms(self):  # pragma: no cover
        path = self.dir_path / '_phy_spikes_subset.waveforms.npy'
        path_channels = self.dir_path / '_phy_spikes_subset.channels.npy'
        path_spikes = self.dir_path / '_phy_spikes_subset.spikes.npy'
        if not path.exists() or not path_channels.exists() or not path_spikes.exists():
            logger.warning(
                "Skipping spike waveforms that do not exist, they will be extracted "
                "on the fly from the raw data as needed.")
            return
        logger.debug("Loading spikes subset waveforms to avoid fetching waveforms from raw data.")
        try:
            return Bunch(
                waveforms=self._read_array(path, mmap_mode='r'),
                spike_channels=self._read_array(path_channels),
                spike_ids=self._read_array(path_spikes),
            )
        except Exception as e:
            logger.warning("Could not load spike waveforms: %s.", e)
            return

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
            path = self._find_path(
                'templates.npy', 'templates.waveforms.npy', 'templates.waveforms.*.npy')
            data = self._read_array(path, mmap_mode='r+')
            data = np.atleast_3d(data)
            assert data.ndim == 3
            assert data.dtype in (np.float32, np.float64)
            # WARNING: this will load the full array in memory, might cause memory problems
            empty_templates = np.all(np.all(np.isnan(data), axis=1), axis=1)
            data[empty_templates, ...] = 0
            n_templates, n_samples, n_channels_loc = data.shape
        except IOError:
            return

        try:
            # WARNING: KS2 saves templates_ind.npy (with an s), and note template_ind.npy,
            # so that file is not taken into account here.
            # That means templates.npy is considered as a dense array.
            # Proper fix would be to save templates.npy as a true sparse array, with proper
            # template_ind.npy (without an s).
            path = self._find_path('template_ind.npy', 'templates.waveformsChannels*.npy')
            cols = self._read_array(path)
            if cols.ndim != 2:  # pragma: no cover
                cols = np.atleast_2d(cols).T
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
        out = np.dot(x, mat) * getattr(self, 'template_scaling', 1.0)
        return np.ascontiguousarray(out)

    def _load_features(self):

        # Sparse structure: regular array with row and col indices.
        try:
            logger.debug("Loading features.")
            data = self._read_array(
                self._find_path('pc_features.npy'), mmap_mode='r')
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
            cols = self._read_array(self._find_path('pc_feature_ind.npy'), mmap_mode='r')
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

    def _find_best_channels(self, template, amplitude_threshold=None):
        """Find the best channels for a given template."""
        # Compute the template amplitude on each channel.
        assert template.ndim == 2  # shape: (n_samples, n_channels)
        amplitude = template.max(axis=0) - template.min(axis=0)
        assert not np.all(np.isnan(amplitude)), "Template is all NaN!"
        assert amplitude.ndim == 1  # shape: (n_channels,)
        # Find the peak channel.
        best_channel = np.argmax(amplitude)
        max_amp = amplitude[best_channel]
        # Find the channels X% peak.
        amplitude_threshold = (
            amplitude_threshold if amplitude_threshold is not None else self.amplitude_threshold)
        peak_channels = np.nonzero(amplitude >= amplitude_threshold * max_amp)[0]
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

    def _template_n_channels(self, template_id, n_channels):
        """Return the n best channels for a given template, filling with -1s if there isn't
        enough best channels for that template."""
        assert n_channels > 0
        if template_id not in self.template_ids:
            return [-1] * n_channels
        template = self.get_template(template_id)
        channel_ids = list(template.channel_ids[:n_channels])
        if len(channel_ids) < n_channels:
            channel_ids += [-1] * (n_channels - len(channel_ids))
        return channel_ids

    def _get_template_dense(self, template_id, channel_ids=None, amplitude_threshold=None,
                            unwhiten=True):
        """Return data for one template."""
        if not self.sparse_templates:
            return
        template_w = self.sparse_templates.data[template_id, ...]
        template = self._unwhiten(template_w).astype(np.float32) if unwhiten else template_w
        assert template.ndim == 2
        channel_ids_, amplitude, best_channel = self._find_best_channels(
            template, amplitude_threshold=amplitude_threshold)
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

    def _get_template_sparse(self, template_id, unwhiten=True):
        data, cols = self.sparse_templates.data, self.sparse_templates.cols
        assert cols is not None
        template_w, channel_ids = data[template_id], cols[template_id]

        # KS2 HACK: dense templates may have been saved as sparse arrays (with all channels),
        # we need to remove channels with no signal.

        # template_w is (n_samples, n_channels)
        template_max = np.abs(template_w).max(axis=0)  # n_channels
        has_signal = template_max > template_max.max() * 1e-6
        channel_ids = channel_ids[has_signal]
        template_w = template_w[:, has_signal]

        # Remove unused channels = -1.
        used = channel_ids != -1
        template_w = template_w[:, used]
        channel_ids = channel_ids[used]
        channel_ids = channel_ids.astype(np.uint32)

        # Unwhiten.
        template = self._unwhiten(template_w, channel_ids=channel_ids) if unwhiten else template_w
        template = template.astype(np.float32)
        assert template.ndim == 2
        assert template.shape[1] == len(channel_ids)
        # Compute the amplitude and the channel with max amplitude.
        amplitude = template.max(axis=0) - template.min(axis=0)
        best_channel = channel_ids[np.argmax(amplitude)]
        # NOTE: it is expected that the channel_ids are reordered by decreasing amplitude.
        # To each column of the template array corresponds the channel id given by channel_ids.
        channels_reordered = np.argsort(amplitude)[::-1]
        out = Bunch(
            template=template[..., channels_reordered],
            amplitude=amplitude,
            best_channel=best_channel,
            channel_ids=channel_ids[channels_reordered],
        )
        return out

    def get_merge_map(self):
        """"Gets the maps of merges and splits between spikes.clusters and spikes.templates"""
        inverse_mapping_dict = {key: [] for key in range(np.max(self.spike_clusters) + 1)}
        for temp in np.unique(self.spike_templates):
            idx = np.where(self.spike_templates == temp)[0]
            new_idx = self.spike_clusters[idx]
            mapping = np.unique(new_idx)
            for n in mapping:
                inverse_mapping_dict[n].append(temp)

        nan_idx = np.array([idx for idx, val in inverse_mapping_dict.items() if len(val) == 0])

        return inverse_mapping_dict, nan_idx

    #--------------------------------------------------------------------------
    # Data access methods
    #--------------------------------------------------------------------------

    def get_template(self, template_id, channel_ids=None, amplitude_threshold=None, unwhiten=True):
        """Get data about a template."""
        if self.sparse_templates and self.sparse_templates.cols is not None:
            return self._get_template_sparse(template_id, unwhiten=unwhiten)
        else:
            return self._get_template_dense(
                template_id, channel_ids=channel_ids, amplitude_threshold=amplitude_threshold,
                unwhiten=unwhiten)

    def get_waveforms(self, spike_ids, channel_ids=None):
        """Return spike waveforms on specified channels."""
        if self.traces is None and self.spike_waveforms is None:
            return
        # Create the output array.
        nsw = self.n_samples_waveforms
        channel_ids = np.arange(self.n_channels) if channel_ids is None else channel_ids

        if self.spike_waveforms is not None:
            # Load from precomputed spikes.
            return get_spike_waveforms(
                spike_ids, channel_ids, spike_waveforms=self.spike_waveforms,
                n_samples_waveforms=nsw)
        else:
            # Or load directly from raw data (slower).
            spike_samples = self.spike_samples[spike_ids]
            return extract_waveforms(
                self.traces, spike_samples, channel_ids, n_samples_waveforms=nsw)

    def get_features(self, spike_ids, channel_ids):
        """Return sparse features for given spikes."""
        sf = self.sparse_features
        if sf is None and self.spike_waveforms is not None:
            ns = len(spike_ids)
            nc = len(channel_ids)
            n_pcs = 3
            features = np.zeros((ns, nc, n_pcs), dtype=np.float32)
            spike_ids_exist = np.intersect1d(spike_ids, self.spike_waveforms.spike_ids)
            # Compute PCs from the waveforms for the spikes that are in spike_waveforms.spike_ids.
            waveforms = self.get_waveforms(spike_ids_exist, channel_ids)
            features_existing = compute_features(waveforms)
            assert features.shape[1:] == (nc, n_pcs)
            # Now we need to integrate the computed features into the output array, knowing
            # that some spikes may be missing if there were requested here in spike_ids, but
            # were absent in spike_waveforms.spike_ids.
            ind = _index_of(spike_ids_exist, spike_ids)
            features[ind, ...] = features_existing
            return features
        elif sf is None:
            return
        _, n_channels_loc, n_pcs = sf.data.shape
        ns = len(spike_ids)
        nc = len(channel_ids)

        # Initialize the output array.
        features = np.empty((ns, n_channels_loc, n_pcs))
        features[:] = np.nan

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

    def get_depths(self):
        """Compute spike depths based on spike pc features and probe depths."""
        # compute the depth as the weighted sum of coordinates
        # if PC features are provided, compute the depth as the weighted sum of coordinates
        nbatch = 50000
        c = 0
        spikes_depths = np.zeros_like(self.spike_times) * np.nan
        nspi = spikes_depths.shape[0]
        if self.sparse_features is None or self.sparse_features.data.shape[0] != self.n_spikes:
            return None
        while True:
            ispi = np.arange(c, min(c + nbatch, nspi))
            # take only first component
            features = self.sparse_features.data[ispi, :, 0]
            features = np.maximum(features, 0) ** 2  # takes only positive values into account
            ichannels = self.sparse_features.cols[self.spike_templates[ispi]].astype(np.uint32)
            # features = np.square(self.sparse_features.data[ispi, :, 0])
            # ichannels = self.sparse_features.cols[self.spike_templates[ispi]].astype(np.int64)
            ypos = self.channel_positions[ichannels, 1]
            with np.errstate(divide='ignore'):
                spikes_depths[ispi] = (np.sum(np.transpose(ypos * features) /
                                              np.sum(features, axis=1), axis=0))
            c += nbatch
            if c >= nspi:
                break
        return spikes_depths

    def get_amplitudes_true(self, sample2unit=1., use='templates'):
        """Convert spike amplitude values to input amplitudes units
         via scaling by unwhitened template waveform.
         :param sample2unit float: factor to convert the raw data to a physical unit (defaults 1.)
         :returns: spike_amplitudes_volts: np.array [nspikes] spike amplitudes in raw data units
         :returns: templates_volts: np.array[ntemplates, nsamples, nchannels]: templates
         in raw data units
         :returns: template_amps_volts: np.array[ntemplates]: average templates amplitudes
          in raw data units
         To scale the template for template matching,
         raw_data_volts = templates_volts * spike_amplitudes_volts / template_amps_volts
         """
        # spike_amp = ks2_spike_amps * maxmin(inv_whitening(ks2_template_amps))
        # to rescale the template,

        if use == 'clusters':
            sparse = self.sparse_clusters
            spikes = self.spike_clusters
            n_wav = self.n_clusters
        else:
            sparse = self.sparse_templates
            spikes = self.spike_templates
            n_wav = self.n_templates

        # unwhiten template waveforms on their channels of max amplitude
        if sparse.cols:
            raise NotImplementedError
        # apply the inverse whitening matrix to the template
        templates_wfs = np.zeros_like(sparse.data)  # nt, ns, nc
        for n in np.arange(n_wav):
            templates_wfs[n, :, :] = np.matmul(sparse.data[n, :, :], self.wmi)

        # The amplitude on each channel is the positive peak minus the negative
        templates_ch_amps = np.max(templates_wfs, axis=1) - np.min(templates_wfs, axis=1)

        # The template arbitrary unit amplitude is the amplitude of its largest channel
        # (but see below for true tempAmps)
        templates_amps_au = np.max(templates_ch_amps, axis=1)
        spike_amps = templates_amps_au[spikes] * self.amplitudes

        with np.errstate(divide='ignore', invalid='ignore'):
            # take the average spike amplitude per template
            templates_amps_v = (np.bincount(spikes, weights=spike_amps) /
                                np.bincount(spikes))
            # scale back the template according to the spikes units
            templates_physical_unit = templates_wfs * (templates_amps_v / templates_amps_au
                                                       )[:, np.newaxis, np.newaxis]

        return (spike_amps * sample2unit,
                templates_physical_unit * sample2unit,
                templates_amps_v * sample2unit)

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

    def get_template_counts(self, cluster_id):
        """Return a histogram of the number of spikes in each template for a given cluster."""
        spike_ids = self.get_cluster_spikes(cluster_id)
        st = self.spike_templates[spike_ids]
        return np.bincount(st, minlength=self.n_templates)

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
        return template.template if template else None

    def get_cluster_mean_waveforms(self, cluster_id, unwhiten=True):
        """Return the mean template waveforms of a cluster, as a weighted average of the
        template waveforms from which the cluster originates from."""
        count = self.get_template_counts(cluster_id)
        best_template = np.argmax(count)
        template_ids = np.nonzero(count)[0]
        count = count[template_ids]
        # Get local channels of the best template for the given cluster.
        template = self.get_template(best_template, unwhiten=unwhiten)
        channel_ids = template.channel_ids
        # Get all templates from which this cluster stems from.
        templates = [self.get_template(template_id, unwhiten=unwhiten)
                     for template_id in template_ids]
        # Construct the waveforms array.
        ns = self.n_samples_waveforms
        data = np.zeros((len(template_ids), ns, self.n_channels))
        for i, b in enumerate(templates):
            data[i][:, b.channel_ids] = b.template
        waveforms = data[..., channel_ids]
        assert waveforms.shape == (len(template_ids), ns, len(channel_ids))
        mean_waveforms = np.average(waveforms, axis=0, weights=count)
        assert mean_waveforms.shape == (ns, len(channel_ids))
        return Bunch(mean_waveforms=mean_waveforms, channel_ids=channel_ids)

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
        """Returns a vector of peak channels for all templates waveforms"""
        return self._channels(self.sparse_templates)

    @property
    def clusters_channels(self):
        """Returns a vector of peak channels for all clusters waveforms"""
        channels = self._channels(self.sparse_clusters)
        return channels

    def _channels(self, sparse):
        """ Gets peak channels for each waveform"""
        tmp = sparse.data
        n_templates, n_samples, n_channels = tmp.shape
        if sparse.cols is None:
            template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
        else:
            # when the templates are sparse, the first channel is the highest amplitude channel
            template_peak_channels = sparse.cols[:, 0]
        assert template_peak_channels.shape == (n_templates,)
        return template_peak_channels

    @property
    def templates_probes(self):
        """Returns a vector of probe index for all templates"""
        return self.channel_probes[self.templates_channels]

    @property
    def templates_amplitudes(self):
        """Returns the average amplitude per cluster"""
        return self._amplitudes(self.spike_templates)

    @property
    def clusters_amplitudes(self):
        """Returns the average amplitude per cluster"""
        return self._amplitudes(self.spike_clusters)

    def _amplitudes(self, tmp):
        """ Compute average amplitude for spikes"""
        tid = np.unique(tmp)
        n = np.bincount(tmp)[tid]
        a = np.bincount(tmp, weights=self.amplitudes)[tid]
        n[np.isnan(n)] = 1
        return a / n

    @property
    def templates_waveforms_durations(self):
        """Returns a vector of waveform durations (ms) for all templates"""
        return self._waveform_durations(self.sparse_templates.data)

    @property
    def clusters_waveforms_durations(self):
        """Returns a vector of waveform durations (ms) for all clusters"""
        waveform_duration = self._waveform_durations(self.sparse_clusters.data)
        return waveform_duration

    def _waveform_durations(self, tmp):
        n_templates, n_samples, n_channels = tmp.shape
        # Compute the peak channels for each template.
        template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
        durations = tmp.argmax(axis=1) - tmp.argmin(axis=1)
        ind = np.ravel_multi_index((np.arange(0, n_templates), template_peak_channels),
                                   (n_templates, n_channels), mode='raise', order='C')
        return durations.flatten()[ind].astype(np.float64) / self.sample_rate * 1e3

    def cluster_waveforms(self):
        """
        Computes the cluster waveforms for split and merged clusters
        :return:
        """
        # Only non sparse implementation
        ns = self.n_samples_waveforms
        data = np.zeros((np.max(self.cluster_ids) + 1, ns, self.n_channels))
        for clust, val in self.merge_map.items():
            if len(val) > 1:
                mean_waveform = self.get_cluster_mean_waveforms(clust, unwhiten=False)
                data[clust, :, mean_waveform.channel_ids] = \
                    np.swapaxes(mean_waveform.mean_waveforms, 0, 1)
            elif len(val) == 1:
                data[clust, :, :] = self.sparse_templates.data[val[0], :, :]

        return Bunch(data=data, cols=None)

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

    def save_spikes_subset_waveforms(self, max_n_spikes_per_template=None, max_n_channels=None,
                                     sample2unit=1.):
        if self.traces is None:
            logger.warning(
                "Spike waveforms could not be extracted as the raw data file is not available.")
            return

        n_chunks_kept = 20  # TODO: better choice
        nst = max_n_spikes_per_template
        nc = max_n_channels or self.n_closest_channels
        nc = max(nc, self.n_closest_channels)

        assert nst > 0
        assert nc > 0

        path = self.dir_path / '_phy_spikes_subset.waveforms.npy'
        path_spikes = self.dir_path / '_phy_spikes_subset.spikes.npy'
        path_channels = self.dir_path / '_phy_spikes_subset.channels.npy'

        # Subselection of spikes.
        spt = _spikes_per_cluster(self.spike_templates)
        template_ids = sorted(spt.keys())
        ss = SpikeSelector(
            get_spikes_per_cluster=lambda cl: spt.get(cl, np.array([], dtype=np.int64)),
            spike_times=self.spike_samples, chunk_bounds=self.traces.chunk_bounds,
            n_chunks_kept=n_chunks_kept)
        spike_ids = ss(max_n_spikes_per_template, template_ids, subset_chunks=True)

        # Save the spike ids.
        ns = len(spike_ids)
        logger.debug("Saving spike waveforms: %d spikes.", ns)
        np.save(path_spikes, spike_ids)

        # Save the spike channels.
        best_channels = np.vstack([
            self._template_n_channels(t, nc) for t in range(self.n_templates)]).astype(np.int32)
        assert best_channels.ndim == 2
        assert best_channels.shape[0] == self.n_templates
        spike_channels = best_channels[self.spike_templates[spike_ids], :]
        assert spike_channels.shape == (ns, nc)
        logger.debug("Saving spike waveforms: spike channels.")
        np.save(path_channels, spike_channels)

        # Extract waveforms from the raw data on a chunk by chunk basis.
        export_waveforms(
            path, self.traces, self.spike_samples[spike_ids], spike_channels,
            n_samples_waveforms=self.n_samples_waveforms, sample2unit=sample2unit)

        # Reload spike waveforms.
        self.spike_waveforms = self._load_spike_waveforms()

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
