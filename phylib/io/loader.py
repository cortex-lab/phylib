# -*- coding: utf-8 -*-

"""Template model loading functions.

Supported file formats:

- KS2
- ALF

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import wraps
import logging
# import os
import os.path as op
from pathlib import Path
# import shutil

import numpy as np
import scipy.io as sio

from phylib.utils import Bunch
from phylib.io.array import _index_of
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
    return params


def _all_positions_distinct(positions):
    """Return whether all positions are distinct."""
    return len(set(tuple(row) for row in positions)) == len(positions)


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
            logger.warning(
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

        assert isinstance(out.cols, np.ndarray)
        assert out.cols.dtype == np.int32
        assert out.cols.ndim == 2
        assert np.all(out.cols >= -1)
        assert out.cols.shape == (n_waveforms, n_channels_loc)

        if 'rows' in out:
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
        n_waveforms, n_channels_loc, n_pcs = out.data.shape

        assert isinstance(out.cols, np.ndarray)
        assert out.cols.dtype == np.int32
        assert out.cols.ndim == 2
        assert np.all(out.cols >= -1)
        assert out.cols.shape == (n_waveforms, n_channels_loc)

        if 'rows' in out:
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

        if 'rows' in out:
            assert isinstance(out.rows, np.ndarray)
            assert out.rows.dtype == np.int32
            assert out.rows.ndim == 1
            assert np.all(out.rows >= -1)
            assert out.rows.shape == (n_spikes,)

        return out
    return wrapped


def validate_amplitudes(f):
    """Amplitudes must be in volts."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.ndim == 1
        assert np.all(out >= 0)
        if np.any(out >= 1):
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


#------------------------------------------------------------------------------
# Computations
#------------------------------------------------------------------------------

@validate_depths
def _compute_spike_depths_from_features(features, spike_templates, channel_pos, batch=50_000):
    ns, nch, nf = features.data.shape
    n_batches = int(np.ceil(ns / float(batch)))
    assert ns > 0
    assert n_batches > 0

    depths = np.zeros(ns, dtype=np.float64)
    for b in range(n_batches):
        spk = slice(b * batch, (b + 1) * batch)
        assert b * batch < ns

        fet = np.maximum(features.data[spk, :, 0], 0) ** 2
        ch = features.cols[spike_templates[spk]] if features.get('cols', None) is not None \
            else np.tile(np.arange(nch), (fet.shape[0], 1))
        ypos = channel_pos[ch, 1]
        assert ypos.shape == (fet.shape[0], nch)

        with np.errstate(all='ignore'):
            d = np.sum(np.transpose(ypos * fet) / np.sum(fet, axis=1), axis=0)
            d[np.isnan(d)] = 0
            depths[spk] = d

    return depths


def _unwhiten_template_waveform(waveform, channels, unw_mat=None, n_channels=None):
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
    amplitude_threshold = .25  # TODO: param
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
    assert waveform_n.shape == (ns, n_channels)
    assert channels.shape == (n_channels,)

    return waveform_n, channels


@validate_waveforms
def _normalize_templates_waveforms(
        waveforms, channels, amplitudes=None, n_channels=None, spike_templates=None,
        unw_mat=None, ampfactor=None):

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
            unw_mat=unw_mat)
        if wc is not None:
            waveforms_n[i, ...], channels_n[i] = wc

    # Convert into physical units
    # ---------------------------
    # The amplitude on each channel is the positive peak minus the negative
    # The template arbitrary unit amplitude is the amplitude of its largest channel

    templates_amps = np.max(
        np.max(waveforms_n, axis=1) - np.min(waveforms_n, axis=1), axis=1)
    spike_amps = templates_amps[spike_templates] * amplitudes
    with np.errstate(all='ignore'):
        # take the average spike amplitude per template
        templates_amps_v = (np.bincount(spike_templates, weights=spike_amps) /
                            np.bincount(spike_templates))
        # scale back the template according to the spikes units
        waveforms_n *= ampfactor * (templates_amps_v / templates_amps)[:, None, None]
    waveforms_n[np.isnan(waveforms_n)] = 0
    assert np.isnan(waveforms_n).sum() == 0

    out = Bunch(
        data=waveforms_n,
        cols=channels,
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

@validate_spike_templates
def _load_spike_templates(spike_templates):
    """Corresponds to spike_templates.npy, spike_clusters.npy, spikes.templates.npy,
    spikes.clusters.npy"""
    return np.asarray(spike_templates, dtype=np.int64).ravel()


# Spike reorder
# -------------

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


@validate_channel_shanks
def _load_channel_shanks(channel_shanks):
    """Corresponds to channel_shanks.npy

    Each probe might have multiple shanks. Shank numbers are *relative* to the each probe.

    """
    return np.asarray(channel_shanks, dtype=np.int32).ravel()


@validate_channel_probes
def _load_channel_probes(channel_probes):
    """Corresponds to channel_probes.npy"""
    return np.asarray(channel_probes, dtype=np.int32).ravel()


# Templates
#----------

@validate_waveforms
def _load_template_waveforms_alf(waveforms, channels):
    waveforms = np.asarray(waveforms, dtype=np.float32)
    waveforms = np.atleast_3d(waveforms)

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

@validate_amplitudes
def _load_amplitudes_alf(amplitudes):
    """Corresponds to spikes.amps.npy, templates.amps.npy. Already in volts."""
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
    wm = np.asarray(wm, dtype=np.float64)
    wm = np.atleast_2d(wm)
    if inverse:
        wmi = wm
        wm = np.linalg.inv(wm)
    else:
        wmi = np.linalg.inv(wm)
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
    def read_params(self):
        params_path = self.data_dir / 'params.py'
        assert params_path.exists()
        return read_params(params_path)

    def open(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        assert self.data_dir.exists()
        assert self.data_dir.is_dir()
        self.params = self.read_params()
        # TODO
        # self.check()

    def check(self):
        """Check consistency of all arrays and model variables."""
        # TODO

    def ar(self, fn, mmap_mode=None):
        return read_array(self.data_dir / fn, mmap_mode=mmap_mode)


class TemplateLoaderKS2(object):
    def _load_spike_times(self):
        return _load_spike_times_ks2(self.ar('spike_times.npy'), self.params.sample_rate)
