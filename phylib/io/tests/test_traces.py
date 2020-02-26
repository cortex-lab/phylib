# -*- coding: utf-8 -*-

"""Testing the BaseEphysTraces class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from numpy.testing import assert_equal as ae
from numpy.testing import assert_allclose as ac
import mtscomp
from pytest import raises, mark, fixture

from phylib.utils import Bunch
from ..traces import (
    _get_subitems, _get_chunk_bounds,
    get_ephys_traces, BaseEphysReader, extract_waveforms, export_waveforms,
    get_spike_waveforms)

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test utils
#------------------------------------------------------------------------------

def test_get_subitems():
    bounds = [0, 2, 5]

    def _a(x, y):
        res = _get_subitems(bounds, x)
        res = [(chk, val.tolist() if isinstance(val, np.ndarray) else val) for chk, val in res]
        assert res == y

    _a(-1, [(1, 2)])
    _a(0, [(0, 0)])
    _a(2, [(1, 0)])
    _a(4, [(1, 2)])
    with raises(IndexError):
        _a(5, [])

    _a(slice(None, None, None), [(0, slice(0, 2, 1)), (1, slice(0, 3, 1))])

    _a(slice(1, None, 1), [(0, slice(1, 2, 1)), (1, slice(0, 3, 1))])

    _a(slice(2, None, 1), [(1, slice(0, 3, 1))])
    _a(slice(3, None, 1), [(1, slice(1, 3, 1))])
    _a(slice(5, None, 1), [])

    _a(slice(0, 4, 1), [(0, slice(0, 2, 1)), (1, slice(0, 2, 1))])
    _a(slice(1, 2, 1), [(0, slice(1, 2, 1))])
    _a(slice(1, -1, 1), [(0, slice(1, 2, 1)), (1, slice(0, 2, 1))])
    _a(slice(-2, -1, 1), [(1, slice(1, 2, 1))])

    _a([0], [(0, [0])])
    _a([2], [(1, [0])])
    _a([4], [(1, [2])])
    with raises(IndexError):
        _a([5], [])

    _a([0, 1], [(0, [0, 1])])
    _a([0, 2], [(0, [0]), (1, [0])])
    _a([0, 3], [(0, [0]), (1, [1])])
    with raises(IndexError):
        _a([0, 5], [(0, [0])])
    _a([3, 4], [(1, [1, 2])])

    _a(([3, 4], None), [(1, [1, 2])])


def test_get_chunk_bounds():
    def _a(x, y, z):
        assert _get_chunk_bounds(x, y) == z

    _a([3], 2, [0, 2, 3])
    _a([3], 3, [0, 3])
    _a([3], 4, [0, 3])

    _a([3, 2], 2, [0, 2, 3, 5])
    _a([3, 2], 3, [0, 3, 5])

    _a([3, 7, 5], 4, [0, 3, 7, 10, 14, 15])
    _a([3, 7, 6], 4, [0, 3, 7, 10, 14, 16])

    _a([3, 7, 5], 10, [0, 3, 10, 15])


#------------------------------------------------------------------------------
# Test ephys reader
#------------------------------------------------------------------------------

sample_rate = 100.


def _iter_traces(tempdir, arr):
    yield arr, dict(sample_rate=sample_rate)

    path = tempdir / 'data.npy'
    np.save(path, arr)
    yield path, dict(sample_rate=sample_rate)

    path = tempdir / 'data.bin'
    with open(path, 'wb') as f:
        arr.tofile(f)
    yield path, dict(sample_rate=sample_rate, dtype=arr.dtype, n_channels=arr.shape[1])

    out = tempdir / 'data.cbin'
    outmeta = tempdir / 'data.ch'
    mtscomp.compress(
        path, out, outmeta, sample_rate=sample_rate,
        n_channels=arr.shape[1], dtype=arr.dtype,
        n_threads=1, check_after_compress=False, quiet=True)
    reader = mtscomp.decompress(out, outmeta, check_after_decompress=False, quiet=True)
    yield reader, {}
    yield out, {}


def test_ephys_reader_1(tempdir):
    arr = np.random.randn(1000, 10)
    for obj, kwargs in _iter_traces(tempdir, arr):
        traces = get_ephys_traces(obj, **kwargs)

        assert isinstance(traces, BaseEphysReader)
        assert traces.dtype == arr.dtype
        assert traces.ndim == 2
        assert traces.shape == arr.shape
        assert traces.n_samples == arr.shape[0]
        assert traces.n_channels == arr.shape[1]

        ac(traces[:], arr)


def test_get_spike_waveforms():
    ns, nsw, nc = 8, 5, 3

    w = np.random.rand(ns, nsw, nc)
    s = np.arange(1, 1 + 2 * ns, 2)
    c = np.tile(np.array([1, 2, 3]), (ns, 1))

    assert w.shape == (ns, nsw, nc)
    assert s.shape == (ns,)
    assert c.shape == (ns, nc)

    sw = Bunch(waveforms=w, spike_ids=s, channel_ids=c)
    out = get_spike_waveforms([5, 1, 3], [2, 1], spike_waveforms=sw, n_samples_waveforms=nsw)

    expected = w[[2, 0, 1], ...][..., [1, 0]]
    ae(out, expected)


def test_waveform_extractor(tempdir):
    data = np.random.randn(2000, 10)
    traces = get_ephys_traces(data, sample_rate=1000)

    nsw = 20
    spike_samples = [5, 25, 100, 1000]
    spike_channels = [[1, 3, 5]] * len(spike_samples)

    export_waveforms(
        tempdir / 'waveforms.npy', traces, spike_samples, spike_channels, n_samples_waveforms=nsw)

    w = np.load(tempdir / 'waveforms.npy')

    # ac(w[2, ...], data[90:110, [1, 3, 5]])
    # ac(w[3, ...], data[990:1010, [1, 3, 5]])
