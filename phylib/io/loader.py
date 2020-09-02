# -*- coding: utf-8 -*-

"""Template model loading functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
import os.path as op
from pathlib import Path
import shutil

import numpy as np

from phylib.utils._misc import read_python

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


#------------------------------------------------------------------------------
# Loading functions
#------------------------------------------------------------------------------

"""
Supported file formats are:

- KS2
- ALF

"""


def _load_spike_times_ks2(spike_samples, sample_rate):
    """Corresponds to spike_times.npy"""
    return np.asarray(spike_samples, dtype=np.float64) / float(sample_rate)


def _load_spike_times_alf(spike_times):
    """Corresponds to spikes.times.npy"""
    return np.asarray(spike_times, dtype=np.float64)


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

    def ar(self, fn, mmap_mode=None):
        return read_array(self.data_dir / fn, mmap_mode=mmap_mode)


class TemplateLoaderKS2(object):
    def _load_spike_times(self):
        return _load_spike_times_ks2(self.ar('spike_times.npy'), self.params.sample_rate)
