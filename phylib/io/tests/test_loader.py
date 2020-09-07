# -*- coding: utf-8 -*-

"""Test template loading functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
# import os
# import os.path as op
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import numpy.random as npr
from numpy.testing import assert_allclose as ac
from pytest import raises

from .conftest import Dataset
from phylib.utils import Bunch
from .. import loader as l

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test computations
#------------------------------------------------------------------------------

def test_compute_spike_depths_from_features():
    ns, nc, nf = 4, 3, 2

    fet = 5 + npr.randn(ns, nc, nf)
    ch = np.tile(np.arange(nc), (ns, 1))
    features = Bunch(data=fet, cols=ch)

    st = [0, 0, 1, 1]
    channel_pos = np.array([[0, 100], [0, 200], [0, 300]])

    features_dense = Bunch(data=fet)
    for batch in (50_000, 2):
        for F in (features, features_dense):
            sd = l._compute_spike_depths_from_features(F, st, channel_pos, batch=batch)
            assert sd.dtype == np.float64
            assert sd.ndim == 1
            assert sd.shape == (ns,)
            assert np.all((10 <= sd) & (sd <= 1000))


def test_normalize_templates_waveforms():
    nt, nw, nc = 3, 4, 2
    w = npr.randn(nt, nw, nc)
    ch = npr.permutation(nt * nc).reshape((nt, nc))
    ns = 6
    amp = [1, 1, 2, 2, 3, 5]
    st = [0, 0, 1, 1, 2, 2]
    tw = l._normalize_templates_waveforms(
        w, ch, amplitudes=amp, n_channels=nc, spike_templates=st)

    assert tw.data.shape == (nt, nw, nc)
    assert tw.cols.shape == (nt, nc)
    assert tw.spike_amps.shape == (ns,)
    assert tw.template_amps.shape == (nt,)

    assert np.all(tw.spike_amps > 0)
    assert np.all(tw.template_amps > 0)


#------------------------------------------------------------------------------
# Test loading helpers
#------------------------------------------------------------------------------

# Spike times
#------------

def test_load_spike_times_ks2():
    ac(l._load_spike_times_ks2([0, 10, 100], 10.), [0, 1, 10])
    ac(l._load_spike_reorder_ks2([0, 10, 100], 10.), [0, 1, 10])


def test_load_spike_times_alf():
    ac(l._load_spike_times_alf([0., 1., 10.]), [0, 1, 10])


def test_validate_spike_times():
    wrong = [[-1, 1], [2, 3, 7, 5]]
    sr = 10
    for st in wrong:
        with raises(ValueError):
            l._load_spike_times_ks2(st, sr)
        with raises(ValueError):
            l._load_spike_times_alf(st)


# Spike templates
#----------------

def test_load_spike_templates():
    ac(l._load_spike_templates([0, 0, 5, -1]), [0, 0, 5, -1])


# Channels
#---------

def test_load_channel_map():
    ac(l._load_channel_map([0, 1, 3, 2]), [0, 1, 3, 2])
    with raises(ValueError):
        l._load_channel_map([0, 1, 2, 2])


def test_load_channel_positions():
    ac(l._load_channel_positions([[0, 0], [1, 0]]), [[0, 0], [1, 0]])
    with raises(ValueError):
        l._load_channel_positions([0, 0, 1, 2])
    with raises(ValueError):
        l._load_channel_positions([[0, 0, 1, 2]])
    # Duplicate channels should not raise an error, but default to a linear probe with
    # an error message.
    ac(l._load_channel_positions([[0, 0], [0, 0]]), [[0, 0], [0, 1]])


def test_load_channel_shanks():
    ac(l._load_channel_shanks([0, 0, 1, 2]), [0, 0, 1, 2])


def test_load_channel_probes():
    ac(l._load_channel_probes([0, 0, 1, 2]), [0, 0, 1, 2])


# Waveforms
# ---------

def test_load_template_waveforms_alf():
    ns, nw, nc = 3, 4, 2
    w = npr.randn(ns, nw, nc)
    ch = npr.permutation(ns * nc).reshape((ns, nc))
    tw = l._load_template_waveforms_alf(w, ch)
    assert tw.data.shape == (ns, nw, nc)
    assert tw.cols.shape == (ns, nc)


def test_load_spike_waveforms():
    ns, nw, nc = 3, 4, 2
    w = npr.randn(ns, nw, nc)
    ch = npr.permutation(ns * nc).reshape((ns, nc))
    tw = l._load_spike_waveforms(w, ch, [2, 3, 5])
    assert tw.data.shape == (ns, nw, nc)
    assert tw.cols.shape == (ns, nc)
    assert tw.rows.shape == (ns,)


# Features
# ---------

def test_load_features():
    ns, nc, nf = 3, 4, 2
    w = npr.randn(ns, nc, nf)
    ch = npr.permutation(ns * nc).reshape((ns, nc))
    fet = l._load_features(w, ch, [2, 3, 5])
    assert fet.data.shape == (ns, nc, nf)
    assert fet.cols.shape == (ns, nc)
    assert fet.rows.shape == (ns,)


def test_load_template_features():
    ns, nc = 3, 4
    w = npr.randn(ns, nc)
    ch = npr.permutation(2 * nc).reshape((2, nc))
    fet = l._load_template_features(w, ch, [2, 3, 5])
    assert fet.data.shape == (ns, nc)
    assert fet.cols.shape == (2, nc)
    assert fet.rows.shape == (ns,)


# Amplitudes
# ----------

def test_load_amplitudes_alf():
    amp = npr.uniform(low=1e-4, high=1e-2, size=10)
    ac(l._load_amplitudes_alf(amp), amp)
    with raises(Exception):
        l._load_amplitudes_alf([-1])


# Depths
# ------

def test_load_depths_alf():
    depths = npr.uniform(low=0, high=1e3, size=10)
    ac(l._load_depths_alf(depths), depths)
    with raises(Exception):
        l._load_depths_alf([-1])


# Whitening matrix
# ----------------

def test_load_whitening_matrix():
    wm0 = npr.randn(5, 5)

    wm, wmi = l._load_whitening_matrix(wm0, inverse=False)
    ac(wm, wm0)
    ac(wm @ wmi, np.eye(5), atol=1e-10)

    wm, wmi = l._load_whitening_matrix(wm0, inverse=True)
    ac(wmi, wm0)
    ac(wm @ wmi, np.eye(5), atol=1e-10)


# Template similarity matrix
# --------------------------

def test_load_similarity_matrix():
    mat0 = npr.randn(5, 5)
    mat = l._load_similarity_matrix(mat0)
    ac(mat, mat0)


#------------------------------------------------------------------------------
# Test loading functions
#------------------------------------------------------------------------------

class TemplateLoaderDenseTests(unittest.TestCase):
    param = 'dense'

    @ classmethod
    def setUpClass(cls):
        cls.ibl = cls.param in ('ks2', 'alf')
        cls.tempdir = Path(tempfile.mkdtemp())
        cls.dset = Dataset(cls.tempdir, cls.param)

    @ classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def test_spike_times(self):
        pass


#------------------------------------------------------------------------------
# Other datasets
#------------------------------------------------------------------------------

class TemplateLoaderSparseTests(TemplateLoaderDenseTests):
    param = 'sparse'


class TemplateLoaderMiscTests(TemplateLoaderDenseTests):
    param = 'misc'


class TemplateLoaderALFTests(TemplateLoaderDenseTests):
    param = 'alf'
