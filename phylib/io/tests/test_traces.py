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
from ..array import _index_of
from ..traces import (
    _get_subitems, _get_chunk_bounds,
    get_ephys_reader, BaseEphysReader, extract_waveforms, export_waveforms, RandomEphysReader,
    _get_spike_ids_rel, _get_spike_waveforms_fast, get_spike_waveforms)

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


def _get_spike_waveforms_reference(
        spike_ids, channel_ids, spike_waveforms=None, n_samples_waveforms=None):
    """Reference implementation of the historical per-spike algorithm."""
    assert spike_waveforms
    assert np.all(np.isin(spike_ids, spike_waveforms.spike_ids))
    spike_ids_rel = _index_of(spike_ids, spike_waveforms.spike_ids)
    ns = len(spike_ids)
    nsw = n_samples_waveforms
    assert nsw > 0
    nc = len(channel_ids)
    assert nc > 0
    out = np.zeros((ns, nsw, nc), dtype=spike_waveforms.waveforms.dtype)
    for i, sid in enumerate(spike_ids_rel):
        ind = spike_waveforms.spike_channels[sid, :]
        channel_common = np.intersect1d(channel_ids, ind)
        if len(channel_ids) > 0:
            cols0 = _index_of(channel_common, channel_ids)
            cols1 = _index_of(channel_common, ind)
            assert len(cols0) == len(cols1)
            out[i, :, cols0] = spike_waveforms.waveforms[sid, :, cols1]
    return out


def _assert_spike_waveforms_match_reference(
        spike_ids, channel_ids, spike_waveforms, n_samples_waveforms):
    expected = _get_spike_waveforms_reference(
        spike_ids, channel_ids, spike_waveforms, n_samples_waveforms)
    actual = get_spike_waveforms(
        spike_ids, channel_ids, spike_waveforms, n_samples_waveforms)
    ae(actual, expected)
    assert actual.dtype == expected.dtype


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


@mark.parametrize('dtype', [np.int16, np.float32, np.float64])
def test_get_spike_waveforms_characterization(dtype):
    """Cover ordering, duplicates, missing channels, and padded sparse rows."""
    nsw = 4
    waveforms = np.arange(4 * nsw * 5, dtype=dtype).reshape(4, nsw, 5)
    spike_waveforms = Bunch(
        waveforms=waveforms,
        spike_ids=np.array([11, 3, 20, 7], dtype=np.int64),
        spike_channels=np.array([
            [5, 2, -1, -1, -1],
            [9, 4, 2, -1, -1],
            [2, 5, 2, 8, -1],
            [8, 7, 6, 5, 4],
        ], dtype=np.int16),
    )

    cases = [
        ([], [2, 5]),
        ([11], [2]),
        ([20, 11, 20, 3], [8, 2, 99, 5]),
        ([7, 3], [9, 4, 2]),
        ([20], [2, 8, 2]),
        ([11, 3], [-1, 2]),
    ]
    for spike_ids, channel_ids in cases:
        _assert_spike_waveforms_match_reference(
            spike_ids, channel_ids, spike_waveforms, nsw)


def test_get_spike_waveforms_noncontiguous():
    rng = np.random.RandomState(0)
    waveforms_base = rng.randn(6, 7, 8).astype(np.float32)
    channels_base = np.tile(np.arange(8, dtype=np.int32), (6, 1))
    spike_waveforms = Bunch(
        waveforms=waveforms_base[:, ::2, ::2],
        spike_ids=np.arange(0, 12, 2, dtype=np.int32),
        spike_channels=channels_base[:, ::2],
    )
    requested_spikes = np.array([8, -99, 0, -99, 4, -99, 8, -99])[::2]
    requested_channels = np.array(
        [6, 0, 0, 0, 13, 0, 2, 0], dtype=np.uint16)[::2]
    assert not spike_waveforms.waveforms.flags.c_contiguous
    assert not spike_waveforms.spike_channels.flags.c_contiguous
    assert not requested_spikes.flags.c_contiguous
    assert not requested_channels.flags.c_contiguous
    _assert_spike_waveforms_match_reference(
        requested_spikes, requested_channels, spike_waveforms, 4)


def test_get_spike_waveforms_memmap(tempdir):
    rng = np.random.RandomState(1)
    waveforms = rng.randn(6, 5, 4).astype(np.float32)
    path = tempdir / 'sparse_waveforms.npy'
    np.save(path, waveforms)
    mapped_waveforms = np.load(path, mmap_mode='r')
    assert isinstance(mapped_waveforms, np.memmap)

    spike_waveforms = Bunch(
        waveforms=mapped_waveforms,
        spike_ids=np.arange(6, dtype=np.int64) * 100_000,
        spike_channels=np.array([
            [8, 3, -1, -1],
            [2, 8, 3, -1],
            [5, 3, 8, 2],
            [8, 8, 1, -1],
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ], dtype=np.int32),
    )
    _assert_spike_waveforms_match_reference(
        [500_000, 0, 300_000, 500_000], [3, 8, 99, 3],
        spike_waveforms, 5)


def test_get_spike_waveforms_randomized():
    rng = np.random.RandomState(42)
    for _ in range(100):
        n_spikes = rng.randint(1, 20)
        n_samples = rng.randint(1, 10)
        n_sparse_channels = rng.randint(1, 9)
        n_requested_spikes = rng.randint(0, 30)
        n_requested_channels = rng.randint(1, 12)
        spike_ids = rng.choice(
            np.arange(100, 100 + 3 * n_spikes), size=n_spikes, replace=False)
        spike_channels = rng.randint(
            -1, 14, size=(n_spikes, n_sparse_channels)).astype(np.int32)
        waveforms = rng.randn(n_spikes, n_samples, n_sparse_channels).astype(np.float32)
        spike_waveforms = Bunch(
            waveforms=waveforms,
            spike_ids=spike_ids,
            spike_channels=spike_channels,
        )
        requested_spikes = rng.choice(
            spike_ids, size=n_requested_spikes, replace=True).tolist()
        requested_channels = rng.randint(
            -1, 18, size=n_requested_channels).astype(np.int64)
        _assert_spike_waveforms_match_reference(
            requested_spikes, requested_channels, spike_waveforms, n_samples)


def test_get_spike_waveforms_empty_channels():
    spike_waveforms = Bunch(
        waveforms=np.zeros((1, 2, 1)),
        spike_ids=np.array([0]),
        spike_channels=np.array([[0]]),
    )
    with raises(AssertionError):
        get_spike_waveforms(
            [0], [], spike_waveforms=spike_waveforms, n_samples_waveforms=2)


def test_get_spike_waveforms_fast_path_structure(monkeypatch):
    spike_waveforms = Bunch(
        waveforms=np.arange(3 * 2 * 3).reshape(3, 2, 3),
        spike_ids=np.array([4, 8, 12]),
        spike_channels=np.array([[3, 1, -1], [2, 3, 1], [1, 1, -1]]),
    )

    def fail_intersect(*args, **kwargs):
        raise AssertionError("the integer fast path called np.intersect1d")

    monkeypatch.setattr(np, 'intersect1d', fail_intersect)
    out = get_spike_waveforms(
        [12, 4, 12], [1, 3, 1], spike_waveforms=spike_waveforms,
        n_samples_waveforms=2)
    assert out.shape == (3, 2, 3)


def test_get_spike_waveforms_fast_path_guard():
    spike_waveforms = Bunch(
        waveforms=np.zeros((1, 2, 2)),
        spike_ids=np.array([0]),
        spike_channels=np.array([[0., 1.]]),
    )
    assert _get_spike_waveforms_fast(
        np.array([0]), [0, 1], spike_waveforms, 2) is None
    spike_waveforms.spike_channels = spike_waveforms.spike_channels.astype(np.int32)
    assert _get_spike_waveforms_fast(
        np.array([[0]]), [0, 1], spike_waveforms, 2) is None
    assert _get_spike_waveforms_fast(
        np.array([0]), [[0, 1]], spike_waveforms, 2) is None


def test_get_spike_waveforms_sparse_global_spike_ids(monkeypatch):
    spike_ids = np.arange(12, dtype=np.int64) * 1_000_000_000 + 3_000_000_123
    spike_waveforms = Bunch(
        waveforms=np.arange(12 * 3 * 2).reshape(12, 3, 2),
        spike_ids=spike_ids,
        spike_channels=np.tile([4, 2], (12, 1)),
    )
    requested = spike_ids[[11, 0, 7, 11, 2]]

    def fail_legacy_resolution(*args, **kwargs):
        raise AssertionError("sorted sparse spike IDs used the dense resolver")

    monkeypatch.setattr('phylib.io.traces._index_of', fail_legacy_resolution)
    monkeypatch.setattr(np, 'isin', fail_legacy_resolution)
    out = get_spike_waveforms(
        requested, [2, 4], spike_waveforms=spike_waveforms,
        n_samples_waveforms=3)
    expected = spike_waveforms.waveforms[[11, 0, 7, 11, 2], :, ::-1]
    ae(out, expected)


def test_get_spike_waveforms_spike_id_resolution_fallback():
    # Duplicate/unsorted lookup IDs use _index_of(), where the last duplicate
    # is selected. A missing requested ID still raises AssertionError.
    available = np.array([30, 10, 30, 20])
    ae(_get_spike_ids_rel([30, 20, 30], available), [2, 3, 2])
    with raises(AssertionError):
        _get_spike_ids_rel([30, 99], available)
    with raises(AssertionError):
        _get_spike_ids_rel([99], np.array([], dtype=np.int64))
    with raises(AssertionError):
        _get_spike_ids_rel([2_000_001], np.array([1, 1_000_001, 3_000_001]))


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
