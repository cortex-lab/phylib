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
from pytest import raises, fixture, mark

from phylib.utils import Bunch
from ..traces import (
    _get_subitems, _get_chunk_bounds,
    get_ephys_reader, BaseEphysReader, extract_waveforms, export_waveforms, RandomEphysReader,
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

@fixture
def arr():
    return np.random.randn(2000, 10)


@fixture(params=[10000, 1000, 100])
def sample_rate(request):
    return request.param


@fixture(params=['numpy', 'npy', 'flat', 'flat_concat', 'mtscomp', 'mtscomp_reader'])
def traces(request, tempdir, arr, sample_rate):
    if request.param == 'numpy':
        return get_ephys_reader(arr, sample_rate=sample_rate)

    elif request.param == 'npy':
        path = tempdir / 'data.npy'
        np.save(path, arr)
        return get_ephys_reader(path, sample_rate=sample_rate)

    elif request.param == 'flat':
        path = tempdir / 'data.bin'
        with open(path, 'wb') as f:
            arr.tofile(f)
        return get_ephys_reader(
            path, sample_rate=sample_rate, dtype=arr.dtype, n_channels=arr.shape[1])

    elif request.param == 'flat_concat':
        path0 = tempdir / 'data0.bin'
        with open(path0, 'wb') as f:
            arr[:arr.shape[0] // 2, :].tofile(f)
        path1 = tempdir / 'data1.bin'
        with open(path1, 'wb') as f:
            arr[arr.shape[0] // 2:, :].tofile(f)
        return get_ephys_reader(
            [path0, path1], sample_rate=sample_rate, dtype=arr.dtype, n_channels=arr.shape[1])

    elif request.param in ('mtscomp', 'mtscomp_reader'):
        path = tempdir / 'data.bin'
        with open(path, 'wb') as f:
            arr.tofile(f)
        out = tempdir / 'data.cbin'
        outmeta = tempdir / 'data.ch'
        mtscomp.compress(
            path, out, outmeta, sample_rate=sample_rate,
            n_channels=arr.shape[1], dtype=arr.dtype,
            n_threads=1, check_after_compress=False, quiet=True)
        reader = mtscomp.decompress(out, outmeta, check_after_decompress=False, quiet=True)
        if request.param == 'mtscomp':
            return get_ephys_reader(reader)
        else:
            return get_ephys_reader(out)


def test_ephys_reader_1(tempdir, arr, traces, sample_rate):
    assert isinstance(traces, BaseEphysReader)
    assert traces.dtype == arr.dtype
    assert traces.ndim == 2
    assert traces.shape == arr.shape
    assert traces.n_samples == arr.shape[0]
    assert traces.n_channels == arr.shape[1]
    assert traces.n_parts in (1, 2)
    assert traces.duration == arr.shape[0] / sample_rate
    assert len(traces.part_bounds) == traces.n_parts + 1
    assert len(traces.chunk_bounds) == traces.n_chunks + 1

    ac(traces[:], arr)

    def _a(f):
        ac(f(traces)[:], f(arr))

    _a(lambda x: x[:, ::-1])

    _a(lambda x: x + 1)
    _a(lambda x: 1 + x)

    _a(lambda x: x - 1)
    _a(lambda x: 1 - x)

    _a(lambda x: x * 2)
    _a(lambda x: 2 * x)

    _a(lambda x: x ** 2)
    _a(lambda x: 2 ** x)

    _a(lambda x: x / 2)
    _a(lambda x: 2 / x)

    _a(lambda x: x / 2.)
    _a(lambda x: 2. / x)

    _a(lambda x: x // 2)
    _a(lambda x: 2 // x)

    _a(lambda x: +x)
    _a(lambda x: -x)

    _a(lambda x: -x[:, [1, 3, 5]])

    _a(lambda x: 1 + x * 2)
    _a(lambda x: 1 + (2 * x))
    _a(lambda x: -x * 2)

    _a(lambda x: x[::1])
    _a(lambda x: x[::1, :])
    _a(lambda x: x[::1, 1:5])
    _a(lambda x: x[::1, ::3])


def test_ephys_random(sample_rate):
    reader = RandomEphysReader(2000, 10, sample_rate=sample_rate)
    assert reader[:10].shape == (10, 10)
    assert reader[:].shape == (2000, 10)
    assert reader[0].shape == (1, 10)
    assert reader[10:20].shape == (10, 10)
    assert reader[[1, 3, 5]].shape == (3, 10)
    assert reader[[1, 3, 5], :].shape == (3, 10)
    assert reader[[1, 3, 5], ::2].shape == (3, 5)
    assert reader[[1, 3, 5], [0, 2, 4]].shape == (3, 3)
    assert reader[0:-1].shape == (1999, 10)
    assert reader[-10:-1].shape == (9, 10)


def test_get_spike_waveforms():
    ns, nsw, nc = 8, 5, 3

    w = np.random.rand(ns, nsw, nc)
    s = np.arange(1, 1 + 2 * ns, 2)
    c = np.tile(np.array([1, 2, 3]), (ns, 1))

    assert w.shape == (ns, nsw, nc)
    assert s.shape == (ns,)
    assert c.shape == (ns, nc)

    sw = Bunch(waveforms=w, spike_ids=s, spike_channels=c)
    out = get_spike_waveforms([5, 1, 3], [2, 1], spike_waveforms=sw, n_samples_waveforms=nsw)

    expected = w[[2, 0, 1], ...][..., [1, 0]]
    ae(out, expected)


@mark.parametrize('do_export', [False, True])
@mark.parametrize('do_cache', [False, True])
def test_waveform_extractor(tempdir, arr, traces, sample_rate, do_export, do_cache):
    data = arr

    nsw = 20
    channel_ids = [1, 3, 5]
    spike_samples = [5, 25, 100, 1000, 1995]
    spike_ids = np.arange(len(spike_samples))
    spike_channels = np.array([channel_ids] * len(spike_samples))

    # Export waveforms into a npy file.
    if do_export:
        export_waveforms(
            tempdir / 'waveforms.npy', traces, spike_samples, spike_channels,
            n_samples_waveforms=nsw, cache=do_cache)
        w = np.load(tempdir / 'waveforms.npy')
    # Extract waveforms directly.
    else:
        w = extract_waveforms(traces, spike_samples, channel_ids, n_samples_waveforms=nsw)

    assert w.dtype == data.dtype == traces.dtype

    spike_waveforms = Bunch(
        spike_ids=spike_ids,
        spike_channels=spike_channels,
        waveforms=w,
    )

    ww = get_spike_waveforms(
        spike_ids, channel_ids, spike_waveforms=spike_waveforms,
        n_samples_waveforms=nsw)
    ae(w, ww)

    assert np.all(w[0, :5, :] == 0)
    ac(w[0, 5:, :], data[0:15, [1, 3, 5]])

    ac(w[1, ...], data[15:35, [1, 3, 5]])
    ac(w[2, ...], data[90:110, [1, 3, 5]])
    ac(w[3, ...], data[990:1010, [1, 3, 5]])

    assert np.all(w[4, -5:, :] == 0)
    ac(w[4, :-5, :], data[-15:, [1, 3, 5]])
