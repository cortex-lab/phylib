# -*- coding: utf-8 -*-

"""Tests of array utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from pytest import raises

from ..array import (
    _unique, _normalize, _index_of, _spikes_in_clusters, _spikes_per_cluster,
    _flatten_per_cluster, get_closest_clusters, _get_data_lim, _flatten, _clip,
    chunk_bounds, excerpts, data_chunk, grouped_mean, SpikeSelector,
    get_excerpts, _range_from_slice, _pad, _get_padded,
    read_array, write_array)
from phylib.utils._types import _as_array
from phylib.utils.testing import _assert_equal as ae
from ..mock import artificial_spike_clusters, artificial_spike_samples


#------------------------------------------------------------------------------
# Test utility functions
#------------------------------------------------------------------------------

def test_clip():
    assert _clip(-1, 0, 1) == 0


def test_range_from_slice():
    """Test '_range_from_slice'."""

    class _SliceTest(object):
        """Utility class to make it more convenient to test slice objects."""
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        def __getitem__(self, item):
            if isinstance(item, slice):
                return _range_from_slice(item, **self._kwargs)

    with raises(ValueError):
        _SliceTest()[:]
    with raises(ValueError):
        _SliceTest()[1:]
    ae(_SliceTest()[:5], [0, 1, 2, 3, 4])
    ae(_SliceTest()[1:5], [1, 2, 3, 4])

    with raises(ValueError):
        _SliceTest()[::2]
    with raises(ValueError):
        _SliceTest()[1::2]
    ae(_SliceTest()[1:5:2], [1, 3])

    with raises(ValueError):
        _SliceTest(start=0)[:]
    with raises(ValueError):
        _SliceTest(start=1)[:]
    with raises(ValueError):
        _SliceTest(step=2)[:]

    ae(_SliceTest(stop=5)[:], [0, 1, 2, 3, 4])
    ae(_SliceTest(start=1, stop=5)[:], [1, 2, 3, 4])
    ae(_SliceTest(stop=5)[1:], [1, 2, 3, 4])
    ae(_SliceTest(start=1)[:5], [1, 2, 3, 4])
    ae(_SliceTest(start=1, step=2)[:5], [1, 3])
    ae(_SliceTest(start=1)[:5:2], [1, 3])

    ae(_SliceTest(length=5)[:], [0, 1, 2, 3, 4])
    with raises(ValueError):
        _SliceTest(length=5)[:3]
    ae(_SliceTest(length=5)[:10], [0, 1, 2, 3, 4])
    ae(_SliceTest(length=5)[:5], [0, 1, 2, 3, 4])
    ae(_SliceTest(start=1, length=5)[:], [1, 2, 3, 4, 5])
    ae(_SliceTest(start=1, length=5)[:6], [1, 2, 3, 4, 5])
    with raises(ValueError):
        _SliceTest(start=1, length=5)[:4]
    ae(_SliceTest(start=1, step=2, stop=5)[:], [1, 3])
    ae(_SliceTest(start=1, stop=5)[::2], [1, 3])
    ae(_SliceTest(stop=5)[1::2], [1, 3])


def test_pad():
    arr = np.random.rand(10, 3)

    ae(_pad(arr, 0, 'right'), arr[:0, :])
    ae(_pad(arr, 3, 'right'), arr[:3, :])
    ae(_pad(arr, 9), arr[:9, :])
    ae(_pad(arr, 10), arr)

    ae(_pad(arr, 12, 'right')[:10, :], arr)
    ae(_pad(arr, 12)[10:, :], np.zeros((2, 3)))

    ae(_pad(arr, 0, 'left'), arr[:0, :])
    ae(_pad(arr, 3, 'left'), arr[7:, :])
    ae(_pad(arr, 9, 'left'), arr[1:, :])
    ae(_pad(arr, 10, 'left'), arr)

    ae(_pad(arr, 12, 'left')[2:, :], arr)
    ae(_pad(arr, 12, 'left')[:2, :], np.zeros((2, 3)))

    with raises(ValueError):
        _pad(arr, -1)


def test_get_padded():
    arr = np.array([1, 2, 3])[:, np.newaxis]

    with raises(RuntimeError):
        ae(_get_padded(arr, -2, 5).ravel(), [1, 2, 3, 0, 0])
    ae(_get_padded(arr, 1, 2).ravel(), [2])
    ae(_get_padded(arr, 0, 5).ravel(), [1, 2, 3, 0, 0])
    ae(_get_padded(arr, -2, 3).ravel(), [0, 0, 1, 2, 3])


def test_get_data_lim():
    arr = np.random.rand(10, 5)
    assert 0 < _get_data_lim(arr) < 1
    assert 0 < _get_data_lim(arr, 2) < 1


def test_unique():
    """Test _unique() function"""
    _unique([])

    n_spikes = 300
    n_clusters = 3
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)
    ae(_unique(spike_clusters), np.arange(n_clusters))


def test_normalize():
    """Test _normalize() function."""

    n_channels = 10
    positions = 1 + 2 * np.random.randn(n_channels, 2)

    # Keep ration is False.
    positions_n = _normalize(positions)

    x_min, y_min = positions_n.min(axis=0)
    x_max, y_max = positions_n.max(axis=0)

    np.allclose(x_min, 0.)
    np.allclose(x_max, 1.)
    np.allclose(y_min, 0.)
    np.allclose(y_max, 1.)

    # Keep ratio is True.
    positions_n = _normalize(positions, keep_ratio=True)

    x_min, y_min = positions_n.min(axis=0)
    x_max, y_max = positions_n.max(axis=0)

    np.allclose(min(x_min, y_min), 0.)
    np.allclose(max(x_max, y_max), 1.)
    np.allclose(x_min + x_max, 1)
    np.allclose(y_min + y_max, 1)


def test_index_of():
    """Test _index_of."""
    arr = [36, 42, 42, 36, 36, 2, 42]
    lookup = _unique(arr)
    ae(_index_of(arr, lookup), [1, 2, 2, 1, 1, 0, 2])


def test_as_array():
    ae(_as_array(3), [3])
    ae(_as_array([3]), [3])
    ae(_as_array(3.), [3.])
    ae(_as_array([3.]), [3.])

    with raises(ValueError):
        _as_array(map)


def test_flatten():
    assert _flatten([[0, 1], [2]]) == [0, 1, 2]


def test_get_closest_clusters():
    out = get_closest_clusters(1, [0, 1, 2], lambda c, d: (d - c))
    assert [_ for _, __ in out] == [2, 1, 0]


#------------------------------------------------------------------------------
# Test read/save
#------------------------------------------------------------------------------

def test_read_write(tempdir):
    arr = np.arange(10).astype(np.float32)

    path = Path(tempdir) / 'test.npy'

    write_array(path, arr)
    ae(read_array(path), arr)
    ae(read_array(path, mmap_mode='r'), arr)


#------------------------------------------------------------------------------
# Test chunking
#------------------------------------------------------------------------------

def test_chunk_bounds():
    chunks = chunk_bounds(200, 100, overlap=20)

    assert next(chunks) == (0, 100, 0, 90)
    assert next(chunks) == (80, 180, 90, 170)
    assert next(chunks) == (160, 200, 170, 200)


def test_chunk():
    data = np.random.randn(200, 4)
    chunks = chunk_bounds(data.shape[0], 100, overlap=20)

    with raises(ValueError):
        data_chunk(data, (0, 0, 0))

    assert data_chunk(data, (0, 0)).shape == (0, 4)

    # Chunk 1.
    ch = next(chunks)
    d = data_chunk(data, ch)
    d_o = data_chunk(data, ch, with_overlap=True)

    ae(d_o, data[0:100])
    ae(d, data[0:90])

    # Chunk 2.
    ch = next(chunks)
    d = data_chunk(data, ch)
    d_o = data_chunk(data, ch, with_overlap=True)

    ae(d_o, data[80:180])
    ae(d, data[90:170])


def test_excerpts_1():
    bounds = [(start, end) for (start, end) in excerpts(100,
                                                        n_excerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10), (45, 55), (90, 100)]


def test_excerpts_2():
    bounds = [(start, end) for (start, end) in excerpts(10,
                                                        n_excerpts=3,
                                                        excerpt_size=10)]
    assert bounds == [(0, 10)]


def test_get_excerpts():
    data = np.random.rand(100, 2)
    subdata = get_excerpts(data, n_excerpts=10, excerpt_size=5)
    assert subdata.shape == (50, 2)
    ae(subdata[:5, :], data[:5, :])
    ae(subdata[-5:, :], data[-10:-5, :])

    data = np.random.rand(10, 2)
    subdata = get_excerpts(data, n_excerpts=10, excerpt_size=5)
    ae(subdata, data)

    data = np.random.rand(10, 2)
    subdata = get_excerpts(data, n_excerpts=1, excerpt_size=10)
    ae(subdata, data)

    assert len(get_excerpts(data, n_excerpts=0, excerpt_size=10)) == 0


#------------------------------------------------------------------------------
# Test spike clusters functions
#------------------------------------------------------------------------------

def test_spikes_in_clusters():
    """Test _spikes_in_clusters()."""

    n_spikes = 100
    n_clusters = 5
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    ae(_spikes_in_clusters(spike_clusters, []), [])

    for i in range(n_clusters):
        assert np.all(spike_clusters[_spikes_in_clusters(spike_clusters, [i])] == i)

    clusters = [1, 2, 3]
    assert np.all(np.in1d(
        spike_clusters[_spikes_in_clusters(spike_clusters, clusters)], clusters))


def test_spikes_per_cluster():
    """Test _spikes_per_cluster()."""

    n_spikes = 100
    n_clusters = 3
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    assert not _spikes_per_cluster([])

    spikes_per_cluster = _spikes_per_cluster(spike_clusters)
    assert list(spikes_per_cluster.keys()) == list(range(n_clusters))

    for i in range(n_clusters):
        ae(spikes_per_cluster[i], np.sort(spikes_per_cluster[i]))
        assert np.all(spike_clusters[spikes_per_cluster[i]] == i)


def test_flatten_per_cluster():
    spc = {2: [2, 7, 11], 3: [3, 5], 5: []}
    arr = _flatten_per_cluster(spc)
    ae(arr, [2, 3, 5, 7, 11])


def test_grouped_mean():
    spike_clusters = np.array([2, 3, 2, 2, 5])
    arr = [9, -3, 10, 11, -5]
    ae(grouped_mean(arr, spike_clusters), [10, -3, -5])


#------------------------------------------------------------------------------
# Test spike selection
#------------------------------------------------------------------------------

def test_select_spikes_1():
    spike_times = np.array([0., 1., 2., 3.3, 4.4])
    spike_clusters = np.array([1, 2, 1, 2, 4])
    chunk_bounds = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    n_chunks_kept = 2
    cluster_ids = [1, 2, 4]
    spikes_ids_kept = [0, 1, 3]

    spc = _spikes_per_cluster(spike_clusters)
    ss = SpikeSelector(
        get_spikes_per_cluster=lambda cl: spc.get(cl, np.array([], dtype=np.int64)),
        spike_times=spike_times, chunk_bounds=chunk_bounds, n_chunks_kept=n_chunks_kept)
    ae(ss.chunks_kept, [0.0, 1.1, 3.3, 4.4])

    ae(ss(3, [], subset_chunks=True), [])
    ae(ss(3, [0], subset_chunks=True), [])
    ae(ss(3, [1], subset_chunks=True), [0])

    ae(ss(None, cluster_ids, subset_chunks=True), spikes_ids_kept)
    ae(ss(0, cluster_ids, subset_chunks=True), spikes_ids_kept)
    ae(ss(3, cluster_ids, subset_chunks=True), spikes_ids_kept)
    ae(ss(2, cluster_ids, subset_chunks=True), spikes_ids_kept)
    assert list(ss(1, cluster_ids, subset_chunks=True)) in [[0, 1], [0, 3]]

    ae(ss(2, cluster_ids, subset_spikes=[0, 1], subset_chunks=True), [0, 1])
    ae(ss(2, cluster_ids, subset_chunks=False), np.arange(5))


def test_select_spikes_2():
    n_spikes = 1000
    n_clusters = 10
    spike_times = artificial_spike_samples(n_spikes)
    spike_times = 10. * spike_times / spike_times.max()
    chunk_bounds = np.linspace(0.0, 10.0, 11)
    n_chunks_kept = 3
    chunks_kept = [0., 1., 4., 5., 8., 9.]
    spike_clusters = artificial_spike_clusters(n_spikes, n_clusters)

    spc = _spikes_per_cluster(spike_clusters)
    ss = SpikeSelector(
        get_spikes_per_cluster=lambda cl: spc.get(cl, np.array([], dtype=np.int64)),
        spike_times=spike_times, chunk_bounds=chunk_bounds, n_chunks_kept=n_chunks_kept)
    ae(ss.chunks_kept, chunks_kept)

    def _check_chunks(sid):
        chunk_ids = np.searchsorted(chunk_bounds, spike_times[sid], 'right') - 1
        ae(np.unique(chunk_ids), [0, 4, 8])

    # Select all spikes belonging to the kept chunks.
    sid = ss(n_spikes, np.arange(n_clusters), subset_chunks=True)
    _check_chunks(sid)

    # Select 10 spikes from each cluster.
    sid = ss(10, np.arange(n_clusters), subset_chunks=True)
    assert np.all(np.diff(sid) > 0)
    _check_chunks(sid)
    ae(np.bincount(spike_clusters[sid]), [10] * 10)
