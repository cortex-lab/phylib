# -*- coding: utf-8 -*-

"""Template model loading functions.

Supported file formats:

- KS2
- ALF

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import wraps, partial
import logging
import os
import os.path as op
from pathlib import Path

import numpy as np
import scipy.io as sio

from phylib.io.array import _index_of
from phylib.io.traces import get_ephys_reader, RandomEphysReader
from phylib.utils import Bunch
from phylib.utils._misc import read_python
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
        logger.debug("Load %s", path)
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


def _make_abs_path(p, dir_path):
    p = Path(p)
    if not op.isabs(p):
        p = dir_path / p
    if not p.exists():
        logger.warning("File %s does not exist.", p)
    return p


def read_params(params_path):
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
    params['ampfactor'] = params.get('ampfactor', 1)
    return Bunch(params)


def _all_positions_distinct(positions):
    """Return whether all positions are distinct."""
    return len(set(tuple(row) for row in positions)) == len(positions)


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


#------------------------------------------------------------------------------
# File format detection
#------------------------------------------------------------------------------

def _which_format(dir_path):
    st_path_ks2 = Path(dir_path) / 'spike_times.npy'
    st_path_alf = Path(dir_path) / 'spikes.times.npy'
    if st_path_ks2.exists():
        return 'ks2'
    elif st_path_alf.exists():
        return 'alf'
    raise IOError("Unknown file format")


def _is_dense(arr_path, ind_path):
    if not ind_path.exists():
        return True
    ind = read_array(ind_path, mmap_mode='r')
    # HACK: even if template_ind.npy exists, it may be trivial (containing all channels)
    # in which case the templates are to be considered dense and not sparse
    if np.allclose(ind - np.arange(ind.shape[1])[np.newaxis, :], 0):
        return True
    return False


def _are_templates_dense(dir_path):
    # only for KS2 datasets
    return _is_dense(
        Path(dir_path) / 'templates.npy', Path(dir_path) / 'templates_ind.npy')


def _are_features_dense(dir_path):
    # only for KS2 datasets
    return _is_dense(
        Path(dir_path) / 'pc_features.npy', Path(dir_path) / 'pc_feature_ind.npy')


def _are_template_features_dense(dir_path):
    # only for KS2 datasets
    return _is_dense(
        Path(dir_path) / 'template_features.npy', Path(dir_path) / 'template_feature_ind.npy')


#------------------------------------------------------------------------------
# Validators
#------------------------------------------------------------------------------

def validate_spike_times(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.ndim == 1
        if not np.all(out >= 0):
            raise ValueError("The spike times must be positive.")
        if not np.all(np.diff(out) >= 0):
            raise ValueError("The spike times must be increasing.")
        return out
    return wrapped


def validate_spike_templates(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int64
        assert out.ndim == 1
        assert np.all(out >= -1)
        uc = np.unique(out)
        if np.max(uc) - np.min(uc) + 1 != uc.size:
            logger.debug(
                "Unreferenced clusters found in templates (generally not a problem)")
        return out
    return wrapped


def validate_channel_map(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        assert out.ndim == 1
        assert np.all(out >= -1)
        if len(np.unique(out)) != len(out):
            raise ValueError("Duplicate channel ids in the channel mapping")
        return out
    return wrapped


def validate_channel_positions(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        if out.ndim != 2:
            raise ValueError("The channel_positions array should have 2 dimensions.")
        if out.shape[1] not in (2, 3):
            raise ValueError("The channel_positions array should have 2/3 columns.")
        if not _all_positions_distinct(out):
            logger.error(
                "Some channels are on the same position, please check the channel positions file.")
            out = linear_positions(out.shape[0])
        return out
    return wrapped


def validate_channel_shanks(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        assert out.ndim == 1
        assert np.all(out >= -1)
        return out
    return wrapped


def validate_channel_probes(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.int32
        assert out.ndim == 1
        assert np.all(out >= -1)
        return out
    return wrapped


def validate_waveforms(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)

        assert isinstance(out.data, np.ndarray)
        assert out.data.dtype in (np.float32, np.float64)
        assert out.data.ndim == 3
        n_waveforms, n_samples, n_channels_loc = out.data.shape

        if out.get('cols', None) is not None:
            assert isinstance(out.cols, np.ndarray)
            assert out.cols.dtype == np.int32
            assert out.cols.ndim == 2
            assert np.all(out.cols >= -1)
            assert out.cols.shape == (n_waveforms, n_channels_loc)

        if out.get('rows', None) is not None:
            assert isinstance(out.rows, np.ndarray)
            assert out.rows.dtype == np.int32
            assert out.rows.ndim == 1
            assert np.all(out.rows >= -1)
            assert out.rows.shape == (n_waveforms,)

        return out
    return wrapped


def validate_features(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)

        assert isinstance(out.data, np.ndarray)
        assert out.data.dtype in (np.float32, np.float64)
        assert out.data.ndim == 3
        n_waveforms, n_pcs, n_channels_loc = out.data.shape

        if out.get('cols', None) is not None:
            assert isinstance(out.cols, np.ndarray)
            assert out.cols.dtype == np.int32
            assert out.cols.ndim == 2
            assert np.all(out.cols >= -1)

        if out.get('rows', None) is not None:
            assert isinstance(out.rows, np.ndarray)
            assert out.rows.dtype == np.int32
            assert out.rows.ndim == 1
            assert np.all(out.rows >= -1)
            assert out.rows.shape == (n_waveforms,)

        return out
    return wrapped


def validate_template_features(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)

        assert isinstance(out.data, np.ndarray)
        assert out.data.dtype in (np.float32, np.float64)
        assert out.data.ndim == 2
        n_spikes, n_channels_loc = out.data.shape

        assert isinstance(out.cols, np.ndarray)
        assert out.cols.dtype == np.int32
        assert out.cols.ndim == 2
        assert np.all(out.cols >= -1)
        # NOTE: the first axis has n_templates rows rather than n_spikes rows
        assert out.cols.shape[1] == n_channels_loc

        if out.get('rows', None) is not None:
            assert isinstance(out.rows, np.ndarray)
            assert out.rows.dtype == np.int32
            assert out.rows.ndim == 1
            assert np.all(out.rows >= -1)
            assert out.rows.shape == (n_spikes,)

        return out
    return wrapped


def validate_amplitudes(f=None, in_volts=True):
    """Amplitudes must be in volts."""
    if f is None:
        return partial(validate_amplitudes, in_volts=in_volts)

    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.ndim == 1
        assert np.all(out >= 0)
        if in_volts and np.any(out >= 1):
            logger.warning("There are %d amplitudes >= 1 volt" % (out >= 1).sum())
        return out
    return wrapped


def validate_depths(f):
    """Depths must be in microns."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.ndim == 1
        assert np.all(out >= 0)
        assert np.all(out <= 1e5)  # in microns
        return out
    return wrapped


def validate_whitening_matrix(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        wm, wmi = f(*args, **kwargs)
        assert isinstance(wm, np.ndarray)
        assert isinstance(wmi, np.ndarray)
        assert wm.ndim == wmi.ndim == 2
        assert wm.shape == wmi.shape
        assert wm.shape[0] == wm.shape[1]
        assert wm.dtype == wmi.dtype == np.float64
        assert np.all(~np.isnan(wm))
        assert np.all(~np.isnan(wmi))
        assert np.allclose(wm @ wmi, np.eye(wm.shape[0]))
        return wm, wmi
    return wrapped


def validate_similarity_matrix(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2
        assert out.shape[0] == out.shape[1]
        assert out.dtype == np.float64
        assert np.all(~np.isnan(out))
        return out
    return wrapped


def ignore_empty_input(f):
    @wraps(f)
    def wrapped(x, *args, **kwargs):
        if x is None:
            logger.debug("Input for %s is None, skipping", f.__name__)
            return None
        return f(x, *args, **kwargs)
    return wrapped


#------------------------------------------------------------------------------
# Computations
#------------------------------------------------------------------------------

@validate_depths
def _compute_spike_depths_from_features(features, spike_templates, channel_pos, batch=50_000):
    ns, nf, nch = features.data.shape
    n_batches = int(np.ceil(ns / float(batch)))
    assert ns > 0
    assert n_batches > 0

    depths = np.zeros(ns, dtype=np.float64)
    for b in range(n_batches):
        spk = slice(b * batch, (b + 1) * batch)
        assert b * batch < ns

        fet = np.maximum(features.data[spk, 0, :], 0) ** 2
        ch = features.cols[spike_templates[spk]] if features.get('cols', None) is not None \
            else np.tile(np.arange(nch), (fet.shape[0], 1))
        ypos = channel_pos[ch, 1]
        assert ypos.shape == (fet.shape[0], nch)

        with np.errstate(all='ignore'):
            d = np.sum(np.transpose(ypos * fet) / np.sum(fet, axis=1), axis=0)
            d[np.isnan(d)] = 0
            depths[spk] = d

    return depths


def _unwhiten_template_waveform(
        waveform, channels, unw_mat=None, n_channels=None,
        amplitude_threshold=None):
    assert n_channels > 0
    ns, nc = waveform.shape
    assert channels.shape == (nc,)
    assert n_channels <= nc

    # Remove unused channels.
    channels_k = channels[channels >= 0]
    nck = len(channels_k)
    assert (nck <= nc)
    if nck == 0:  # pragma: no cover
        return None
    waveform_n = waveform[:, _index_of(channels_k, channels)]
    channels = channels_k

    # Unwhitening
    mat = unw_mat[np.ix_(channels_k, channels_k)]
    assert np.sum(np.isnan(mat.ravel())) == 0
    assert waveform.shape[1] == mat.shape[0] == mat.shape[1]
    waveform_n = waveform_n @ mat
    assert waveform_n.shape[1] == mat.shape[0]

    # Select the channels with signal.
    # HACK: transpose is a work-around this NumPy issue
    # https://stackoverflow.com/a/35020886/1595060
    # amplitude_threshold = .25 if amplitude_threshold is None else amplitude_threshold
    amplitude_threshold = 0  # DEBUG
    amplitude = waveform_n.max(axis=0) - waveform_n.min(axis=0)
    assert amplitude.shape == (nck,)
    assert np.all(amplitude >= 0)
    has_signal = amplitude > amplitude.max() * amplitude_threshold
    if has_signal.sum() == 0:
        return None

    # Remove channels with no signal.
    channels_k = channels_k[has_signal]
    nck = len(channels_k)
    amplitude = amplitude[has_signal]
    assert amplitude.shape == (nck,)
    if nck == 0:  # pragma: no cover
        return None

    # Reorder the channels by decreasing amplitude.
    reorder = np.argsort(amplitude)[::-1]
    assert reorder.shape == (nck,)
    channels_k = channels_k[reorder]
    assert channels_k.shape == (nck,)
    waveform_n = waveform_n[:, _index_of(channels_k, channels)]
    assert np.all(np.diff(waveform_n.max(axis=0) - waveform_n.min(axis=0)) <= 0)

    # Keep the requested number of channels.
    waveform_n = waveform_n[:, :n_channels]
    channels_k = channels_k[:n_channels]
    assert waveform_n.shape[0] == ns
    assert waveform_n.shape[1] == channels_k.shape[0]
    # Pad arrays with zeros if not enough channels.
    if waveform_n.shape[1] < n_channels:
        waveform_n = np.hstack((waveform_n, np.zeros(
            (ns, n_channels - waveform_n.shape[1]), dtype=waveform_n.dtype)))
    if channels_k.shape[0] < n_channels:
        channels_k = np.hstack((channels_k, -np.ones(
            (n_channels - channels_k.shape[0]), dtype=channels_k.dtype)))
    assert waveform_n.shape == (ns, n_channels)
    assert channels_k.shape == (n_channels,)

    return waveform_n, channels_k


@validate_waveforms
def _normalize_templates_waveforms(
        waveforms, channels, amplitudes=None, n_channels=None, spike_templates=None,
        unw_mat=None, ampfactor=None, amplitude_threshold=None):

    # Input validation.
    if not ampfactor:
        logger.warning("No ampfactor provided, conversion to physical units impossible")
        ampfactor = 1
    assert n_channels > 0
    if unw_mat is None:
        logger.warning(
            "No unwhitening matrix provided")
        unw_mat = np.eye(channels.max() + 1)
    assert amplitudes is not None
    assert spike_templates is not None
    waveforms = np.asarray(waveforms, dtype=np.float32)
    nt, ns, nc = waveforms.shape
    channels = np.asarray(channels, dtype=np.int32)
    assert channels.shape == (nt, nc)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    spike_templates = np.asarray(spike_templates, dtype=np.int32)
    assert amplitudes.shape == spike_templates.shape

    # Create the output arrays.
    waveforms_n = np.zeros((nt, ns, n_channels), dtype=np.float32)
    channels_n = np.zeros((nt, n_channels), dtype=np.int32)

    # Unwhitening
    # -----------
    # Unwhiten all templates, select the channels with some signal, and reorder the channels
    # by decreasing waveform amplitude.
    for i in range(nt):
        wc = _unwhiten_template_waveform(
            waveforms[i], channels[i], n_channels=n_channels,
            unw_mat=unw_mat, amplitude_threshold=amplitude_threshold)
        if wc is not None:
            w, c = wc
            assert w.shape == (ns, n_channels)
            assert c.shape == (n_channels,)
            waveforms_n[i, :, :], channels_n[i, :] = w, c

    # Convert into physical units
    # ---------------------------
    # The amplitude on each channel is the positive peak minus the negative
    # The template arbitrary unit amplitude is the amplitude of its largest channel

    templates_amps = np.max(
        np.max(waveforms_n, axis=1) - np.min(waveforms_n, axis=1), axis=1)
    spike_amps = templates_amps[spike_templates] * amplitudes
    spike_amps[np.isnan(spike_amps)] = 0
    with np.errstate(all='ignore'):
        # take the average spike amplitude per template
        templates_amps_v = (
            np.bincount(spike_templates, minlength=len(templates_amps), weights=spike_amps) /
            np.bincount(spike_templates, minlength=len(templates_amps)))

        # scale back the template according to the spikes units
        waveforms_n *= ampfactor * (templates_amps_v / templates_amps)[:, None, None]

    waveforms_n[np.isnan(waveforms_n)] = 0
    assert np.isnan(waveforms_n).sum() == 0
    templates_amps_v[np.isnan(templates_amps_v)] = 0

    out = Bunch(
        data=waveforms_n,
        cols=channels_n,
        spike_amps=spike_amps * ampfactor,
        template_amps=templates_amps_v * ampfactor,
    )
    return out


#------------------------------------------------------------------------------
# Loading functions
#------------------------------------------------------------------------------

# Spike times
#------------

@validate_spike_times
def _load_spike_times_ks2(spike_samples, sample_rate):
    """Corresponds to spike_times.npy"""
    return np.asarray(spike_samples, dtype=np.float64).ravel() / float(sample_rate)


@validate_spike_times
def _load_spike_times_alf(spike_times):
    """Corresponds to spikes.times.npy"""
    return np.asarray(spike_times, dtype=np.float64).ravel()


# Spike templates
#----------------

@ignore_empty_input
@validate_spike_templates
def _load_spike_templates(spike_templates):
    """Corresponds to spike_templates.npy, spike_clusters.npy, spikes.templates.npy,
    spikes.clusters.npy"""
    return np.asarray(spike_templates, dtype=np.int64).ravel()


# Spike reorder
# -------------

@ignore_empty_input
def _load_spike_reorder_ks2(spike_reorder, sample_rate):
    return _load_spike_times_ks2(spike_reorder, sample_rate)


# Channels
#---------

@validate_channel_map
def _load_channel_map(channel_map):
    """Corresponds to channel_map.npy, channels.rawInd.npy"""
    return np.asarray(channel_map, dtype=np.int32).ravel()


@validate_channel_positions
def _load_channel_positions(channel_positions):
    """Corresponds to channel_positions.npy"""
    return np.atleast_2d(np.asarray(channel_positions, dtype=np.float64))


@ignore_empty_input
@validate_channel_shanks
def _load_channel_shanks(channel_shanks):
    """Corresponds to channel_shanks.npy

    Each probe might have multiple shanks. Shank numbers are *relative* to the each probe.

    """
    return np.asarray(channel_shanks, dtype=np.int32).ravel()


@ignore_empty_input
@validate_channel_probes
def _load_channel_probes(channel_probes):
    """Corresponds to channel_probes.npy"""
    return np.asarray(channel_probes, dtype=np.int32).ravel()


# Templates
#----------

@validate_waveforms
def _load_template_waveforms(waveforms, channels):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    waveforms = np.atleast_3d(waveforms)

    if channels is not None:
        channels = np.asarray(channels, dtype=np.int32)
        channels = np.atleast_2d(channels)
        if channels.ndim != 2:  # pragma: no cover
            channels = channels.T

    return Bunch(data=waveforms, cols=channels)


@validate_waveforms
def _load_spike_waveforms(waveforms, channels, spikes):
    waveforms = np.asarray(waveforms, dtype=np.float64)
    waveforms = np.atleast_3d(waveforms)

    channels = np.asarray(channels, dtype=np.int32)
    channels = np.atleast_2d(channels)
    if channels.ndim != 2:  # pragma: no cover
        channels = channels.T

    spikes = np.asarray(spikes, dtype=np.int32)
    return Bunch(data=waveforms, cols=channels, rows=spikes)


# Features
#----------

@validate_features
def _load_features(features, channels=None, spikes=None):
    features = np.asarray(features, dtype=np.float32)
    features = np.atleast_3d(features)

    if channels is not None:
        channels = np.asarray(channels, dtype=np.int32)
        channels = np.atleast_2d(channels)

    if spikes is not None:
        spikes = np.asarray(spikes, dtype=np.int32)

    return Bunch(data=features, cols=channels, rows=spikes)


@validate_template_features
def _load_template_features(features, channels=None, spikes=None):
    features = np.asarray(features, dtype=np.float32)
    features = np.atleast_2d(features)

    if channels is not None:
        channels = np.asarray(channels, dtype=np.int32)
        channels = np.atleast_2d(channels)

    if spikes is not None:
        spikes = np.asarray(spikes, dtype=np.int32)

    return Bunch(data=features, cols=channels, rows=spikes)


# Amplitudes
# ----------

@validate_amplitudes(in_volts=True)
def _load_amplitudes_alf(amplitudes):
    """Corresponds to spikes.amps.npy, templates.amps.npy. Already in volts."""
    return np.asarray(amplitudes, dtype=np.float64).ravel()


@validate_amplitudes(in_volts=False)
def _load_amplitudes_ks2(amplitudes):
    """Corresponds to amplitudes.npy."""
    return np.asarray(amplitudes, dtype=np.float64).ravel()


# Depths
# ------

@validate_depths
def _load_depths_alf(depths):
    """Corresponds to spikes.depths.npy, templates.depths.npy. In microns."""
    return np.asarray(depths, dtype=np.float64).ravel()


# Whitening matrix
# ----------------

@validate_whitening_matrix
def _load_whitening_matrix(wm, inverse=None):
    if wm is None:
        return None, None
    wm = np.asarray(wm, dtype=np.float64)
    wm = np.atleast_2d(wm)
    if inverse:
        wmi = wm
        wm = np.linalg.inv(wm)
    else:
        wmi = np.linalg.inv(wm)
    if np.all(wm == np.eye(wm.shape[0])) or np.all(wmi == np.eye(wmi.shape[0])):
        logger.warning("the whitening matrix is the identity!")
    return wm, wmi


# Similarity matrix
# -----------------

@validate_similarity_matrix
def _load_similarity_matrix(mat):
    mat = np.asarray(mat, dtype=np.float64)
    mat = np.atleast_2d(mat)
    return mat


#------------------------------------------------------------------------------
# Loading classes
#------------------------------------------------------------------------------

class BaseTemplateLoader(object):
    """The Loader provides the data as a set of in-memory arrays in the right physical units.

    The model provides high-level data access methods that leverage these arrays in the loader.

    There should be 1 loader per data format/variant.

    """
    spike_times = None
    spike_times_reorder = None
    spike_templates = None
    spike_clusters = None
    spike_depths = None
    spike_amps = None
    ks2_amplitudes = None

    templates = None
    template_amps = None
    similar_templates = None

    channel_map = None
    channel_positions = None
    channel_shanks = None
    channel_probes = None

    wmi = None
    wm = None

    features = None
    template_features = None

    traces = None

    def load_params(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        assert self.data_dir.exists()
        assert self.data_dir.is_dir()
        params_path = self.data_dir / 'params.py'
        assert params_path.exists()
        self.params = read_params(params_path)

    def load_traces(self):
        if not self.params.dat_path:
            if os.environ.get('PHY_VIRTUAL_RAW_DATA', None):  # pragma: no cover
                n_samples = int((self.spike_times[-1] + 1) * self.sample_rate)
                return RandomEphysReader(
                    n_samples, len(self.channel_map), sample_rate=self.sample_rate)
            return
        # self.dat_path could be any object accepted by get_ephys_reader().
        traces = get_ephys_reader(
            self.params.dat_path, n_channels_dat=self.params.n_channels_dat,
            dtype=self.params.dtype, offset=self.params.offset,
            sample_rate=self.params.sample_rate, ampfactor=self.params.ampfactor)
        if traces is not None:
            traces = traces[:, self.channel_map]  # lazy permutation on the channel axis
        self.traces = traces

    def check(self):
        """Check consistency of all arrays and model variables."""
        # Check spike times
        assert np.all(self.spike_times >= 0)
        assert np.all(np.diff(self.spike_times) >= 0)
        ns = len(self.spike_times)

        if self.spike_times_reorder is not None:
            assert len(self.spike_times_reorder) == ns
        assert len(self.spike_templates) == ns
        assert len(self.spike_clusters) == ns

        nc = len(self.channel_map)
        assert self.channel_positions.shape == (nc, 2)
        if self.channel_shanks is not None:
            assert len(self.channel_shanks) == nc
        if self.channel_probes is not None:
            assert len(self.channel_probes) == nc

        assert self.wm.shape == self.wmi.shape == (nc, nc)

        nt = self.templates.data.shape[0]

        if self.similar_templates is not None:
            assert self.similar_templates.shape == (nt, nt)

        if self.features is not None:
            assert self.features.data.shape[0] == ns
        if self.template_features is not None:
            assert self.template_features.data.shape[0] == ns

        assert self.spike_amps.shape == (ns,)
        assert self.template_amps.shape == (nt,)

    def ar(self, fn, mmap_mode=None, mandatory=True, default=None):
        if isinstance(fn, (tuple, list)):
            # Handle list of files, take the first that exists.
            for fn_ in fn:
                out = self.ar(fn_, mmap_mode=mmap_mode, mandatory=False, default=None)
                if out is not None:
                    return out
            fn = fn[0]
        path = self.data_dir / fn
        if path.exists():
            return read_array(path, mmap_mode=mmap_mode)
        if mandatory:
            raise IOError("File %s does not exist" % fn)
        # File does not exist.
        return default


class TemplateLoaderKS2(BaseTemplateLoader):
    MAX_N_CHANNELS_TEMPLATES = 32

    def check(self):
        super(TemplateLoaderKS2, self).check()
        ns = len(self.spike_times)
        assert len(self.ks2_amplitudes) == ns

    def open(self, data_dir):
        self.load_params(data_dir)
        self.load_traces()

        sr = self.params.sample_rate

        # Spike times.
        self.spike_times = _load_spike_times_ks2(self.ar('spike_times.npy'), sr)
        assert self.spike_times.ndim == 1
        self.n_spikes = len(self.spike_times)

        self.spike_times_reorder = _load_spike_reorder_ks2(
            self.ar('spike_times_reordered.npy', mandatory=False), sr)

        # Spike templates and clusters.
        self.spike_templates = _load_spike_templates(self.ar('spike_templates.npy'))
        self.spike_clusters = _load_spike_templates(
            self.ar('spike_clusters.npy', mandatory=False, default=self.spike_templates))

        # KS2 amplitudes.
        self.ks2_amplitudes = _load_amplitudes_ks2(self.ar('amplitudes.npy'))

        # Channel informations.
        self.channel_map = _load_channel_map(self.ar('channel_map.npy'))
        nc = self.channel_map.shape[0]
        self.n_channels = nc
        self.channel_positions = _load_channel_positions(self.ar('channel_positions.npy'))
        self.channel_shanks = _load_channel_shanks(self.ar('channel_shanks.npy', mandatory=False))
        self.channel_probes = _load_channel_probes(self.ar('channel_probes.npy', mandatory=False))

        # Whitening matrix and its inverse.
        self.wm, self.wmi = _load_whitening_matrix(
            self.ar('whitening_mat.npy', mandatory=False))
        if self.wm is None:
            self.wmi, self.wm = _load_whitening_matrix(
                self.ar('whitening_mat_inv.npy', mandatory=False, default=np.eye(nc)),
                inverse=True)
        assert self.wm is not None and self.wmi is not None
        assert self.wm.shape == (nc, nc)
        assert np.allclose(self.wm @ self.wmi, np.eye(nc))
        assert self.wmi.shape == (nc, nc)

        # Similar templates.
        self.similar_templates = _load_similarity_matrix(self.ar('similar_templates.npy'))

        # PC features.
        self.features = _load_features(
            self.ar('pc_features.npy'),
            channels=self.ar('pc_feature_ind.npy', mandatory=False))

        # Template features.
        self.template_features = _load_template_features(
            self.ar('template_features.npy'),
            self.ar('template_feature_ind.npy', mandatory=False))

        # Compute amplitudes and depths in physical units.
        self.spike_depths = _compute_spike_depths_from_features(
            self.features, self.spike_templates, self.channel_positions, batch=50_000)

        # Template waveforms.
        self.templates = _normalize_templates_waveforms(
            self.ar('templates.npy'),
            self.ar(('template_ind.npy', 'templates_ind.npy'), mandatory=False),
            amplitudes=self.ks2_amplitudes,
            n_channels=self.MAX_N_CHANNELS_TEMPLATES,
            spike_templates=self.spike_templates,
            unw_mat=self.wmi,
            ampfactor=self.params.ampfactor,
            # NOTE: need 0 here to avoid destroying data in temp waveforms and to ensure
            # consistency with previously-saved ALF files
            amplitude_threshold=0,
        )

        # Get the spike and template amplitudes.
        self.spike_amps = self.templates.pop('spike_amps')
        self.template_amps = self.templates.pop('template_amps')

        self.check()


class TemplateLoaderAlf(BaseTemplateLoader):
    def open(self, data_dir):
        self.load_params(data_dir)
        self.load_traces()

        # Spike times.
        self.spike_times = _load_spike_times_alf(self.ar('spikes.times.npy'))
        assert self.spike_times.ndim == 1
        self.n_spikes = len(self.spike_times)

        # Spike templates and clusters.
        self.spike_templates = _load_spike_templates(self.ar('spikes.templates.npy'))
        self.spike_clusters = _load_spike_templates(
            self.ar('spikes.clusters.npy', mandatory=False, default=self.spike_templates))

        # KS2 amplitudes.
        self.spike_amps = _load_amplitudes_alf(self.ar('spikes.amps.npy'))
        self.template_amps = _load_amplitudes_alf(self.ar('templates.amps.npy'))

        # Channel informations.
        self.channel_map = _load_channel_map(self.ar('channels.rawInd.npy'))
        nc = self.channel_map.shape[0]
        self.n_channels = nc
        self.channel_positions = _load_channel_positions(self.ar('channels.localCoordinates.npy'))
        self.channel_shanks = _load_channel_shanks(self.ar('channels.shanks.npy', mandatory=False))
        self.channel_probes = _load_channel_probes(self.ar('channels.probes.npy', mandatory=False))

        # Whitening matrix and its inverse.
        I = np.eye(nc)
        self.wm, self.wmi = _load_whitening_matrix(
            self.ar('_kilosort_whitening.matrix.npy', mandatory=False, default=I))
        assert self.wm is not None and self.wmi is not None
        assert self.wm.shape == (nc, nc)
        assert np.allclose(self.wm @ self.wmi, I)
        assert self.wmi.shape == (nc, nc)

        # Templates.
        self.templates = _load_template_waveforms(
            self.ar('templates.waveforms.npy'),
            self.ar('templates.waveformsChannels.npy', mandatory=False))

        # Compute amplitudes and depths in physical units.
        self.spike_depths = _load_depths_alf(self.ar('spikes.depths.npy'))

        # Spike waveforms subset.
        self.spike_waveforms = _load_spike_waveforms(
            self.ar('_phy_spikes_subset.waveforms.npy', mandatory=False),
            self.ar('_phy_spikes_subset.channels.npy', mandatory=False),
            self.ar('_phy_spikes_subset.spikes.npy', mandatory=False),
        )

        # The following objects are not extracted in ALF and may need to be computed on the fly
        # in the model:
        #   - Similar templates
        #   - PC features
        #   - Template features

        self.check()
