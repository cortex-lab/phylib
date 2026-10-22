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
from pytest import raises, fixture, mark, importorskip

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

    traces.prefetch(0)
    traces.cancel_prefetch()
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
    class CountingMemmap(np.memmap):
        reads = 0

        def __getitem__(self, item):
            type(self).reads += 1
            return super().__getitem__(item)

    rng = np.random.RandomState(1)
    waveforms = rng.randn(6, 5, 4).astype(np.float32)
    path = tempdir / 'sparse_waveforms.npy'
    np.save(path, waveforms)
    mapped_waveforms = np.load(path, mmap_mode='r').view(CountingMemmap)
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
    CountingMemmap.reads = 0
    get_spike_waveforms(
        [500_000, 0, 300_000, 500_000], [3, 8, 99, 3],
        spike_waveforms, 5)
    assert CountingMemmap.reads == 1


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


#------------------------------------------------------------------------------
# Test SpikeInterface reader
#------------------------------------------------------------------------------

@fixture
def si_recording(tempdir):
    si = importorskip('spikeinterface')
    arr = np.random.randint(-1000, 1000, size=(2345, 7)).astype(np.int16)
    arr.tofile(tempdir / 'traces.raw')
    recording = si.read_binary(
        tempdir / 'traces.raw', sampling_frequency=1000., dtype=arr.dtype,
        num_channels=arr.shape[1])
    return recording, arr


def _check_si_traces(traces, arr):
    assert traces.shape == arr.shape
    assert traces.dtype == arr.dtype
    assert traces.sample_rate == 1000.
    ae(traces[:], arr)
    ae(traces[290:910], arr[290:910])
    ae(traces[2300:], arr[2300:])
    ae(traces[[5, 299, 300, 2344]], arr[[5, 299, 300, 2344]])
    ae(traces[[2344, 5, 17]], arr[[2344, 5, 17]])
    ae(traces[17], arr[17:18])
    ae(traces[100:700, [6, 0, 3]], arr[100:700][:, [6, 0, 3]])
    ae((traces * 2)[:50], arr[:50] * 2)


def test_spikeinterface_reader(tempdir, si_recording):
    from ..traces import SpikeInterfaceEphysReader
    recording, arr = si_recording
    path = tempdir / 'recording.json'
    recording.dump_to_json(path)

    # 300-sample blocks, so reads span several blocks and the last one is partial.
    for obj in (recording, path, [path]):
        traces = get_ephys_reader(obj, block_duration=.3)
        assert isinstance(traces, SpikeInterfaceEphysReader)
        _check_si_traces(traces, arr)
        traces.close()
        assert not traces._cache


def test_spikeinterface_reader_relative_json(tempdir, si_recording):
    recording, arr = si_recording
    path = tempdir / 'recording.json'
    recording.dump_to_json(path, relative_to=tempdir)
    _check_si_traces(get_ephys_reader(path, block_duration=.3), arr)


def test_spikeinterface_reader_zarr(tempdir, si_recording):
    importorskip('zarr')
    recording, arr = si_recording
    recording.save(folder=tempdir / 'recording.zarr', format='zarr')
    _check_si_traces(get_ephys_reader(tempdir / 'recording.zarr', block_duration=.3), arr)


def test_spikeinterface_reader_not_recording(tempdir, si_recording):
    import spikeinterface as si
    from probeinterface import generate_linear_probe
    from ..traces import SpikeInterfaceEphysReader, ArrayEphysReader, _get_ephys_constructor
    recording, _ = si_recording
    probe = generate_linear_probe(num_elec=recording.get_num_channels())
    probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
    recording.set_probe(probe, in_place=True)
    sorting = si.NumpySorting.from_unit_dict({0: np.array([10, 20])}, sampling_frequency=1000.)
    with raises(TypeError):
        SpikeInterfaceEphysReader(sorting)
    if hasattr(si, 'create_sorting_analyzer'):  # spikeinterface >= 0.101
        analyzer = si.create_sorting_analyzer(
            sorting, recording, format='zarr', folder=tempdir / 'analyzer.zarr', sparse=False)
        # SpikeInterface < 0.102 loads only extractors, so it fails on an analyzer itself.
        with raises((TypeError, ValueError)):
            get_ephys_reader(analyzer.folder)
    # Only recordings are dispatched to the SpikeInterface reader.
    assert _get_ephys_constructor(sorting)[0] is ArrayEphysReader


def test_spikeinterface_reader_multi_segment():
    si = importorskip('spikeinterface')
    arr = np.zeros((10, 2), dtype=np.int16)
    recording = si.NumpyRecording([arr, arr], sampling_frequency=1000.)
    with raises(ValueError):
        get_ephys_reader(recording)


def test_spikeinterface_reader_cache(si_recording):
    recording, arr = si_recording
    block_bytes = 300 * arr.shape[1] * arr.itemsize
    traces = get_ephys_reader(
        recording, block_duration=.3, cache_size=2 * block_bytes)
    ae(traces[:], arr)
    assert list(traces._cache) == [6, 7]
    ae(traces[:10], arr[:10])
    assert list(traces._cache) == [7, 0]

    # Clones made by lazy operations share the cache.
    clone = traces[:, [1, 2]]
    ae(clone[650:660], arr[650:660, [1, 2]])
    assert clone._cache is traces._cache
    assert list(traces._cache) == [0, 2]


def test_spikeinterface_reader_sparse_index(si_recording):
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.3)
    ae(traces[[5, 2344]], arr[[5, 2344]])
    assert sorted(traces._cache) == [0, 7]


def test_spikeinterface_reader_params_mismatch(si_recording, caplog):
    recording, arr = si_recording
    with caplog.at_level(logging.WARNING, logger='phylib.io.traces'):
        get_ephys_reader(recording, sample_rate=1000., n_channels_dat=7, dtype='int16', offset=0)
    assert not caplog.records
    with caplog.at_level(logging.WARNING, logger='phylib.io.traces'):
        traces = get_ephys_reader(
            recording, sample_rate=30000., n_channels_dat=384, dtype=np.float32, offset=0)
    messages = ' '.join(r.getMessage() for r in caplog.records)
    assert all(name in messages for name in ('sample_rate', 'n_channels', 'dtype'))
    # The recording still sets the values.
    assert (traces.sample_rate, traces.n_channels, traces.dtype) == (1000., 7, np.int16)


class _SlowRecording(object):
    """Stands in for a recording whose get_traces blocks, fails or counts calls."""

    def __init__(self, recording, gate=None, fail=()):
        import threading
        self.recording = recording
        self.gate = gate
        self.fail = set(fail)
        self.calls = []
        self.started = threading.Event()

    def get_traces(self, start_frame=None, end_frame=None, segment_index=0):
        self.calls.append(start_frame)
        self.started.set()
        if self.gate is not None and start_frame in self.gate:
            assert self.gate[start_frame].wait(5), "Timed out waiting for test gate"
        if start_frame in self.fail:
            self.fail.discard(start_frame)
            raise OSError("transient")
        return self.recording.get_traces(
            start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)


def _wait_prefetches(traces):
    blocks = traces._blocks
    with blocks._condition:
        assert blocks._condition.wait_for(
            lambda: blocks._hint is None and blocks._active is None, timeout=5)


def test_spikeinterface_reader_prefetch(si_recording):
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.3)
    traces.recording = slow = _SlowRecording(recording)

    # Ordinary reads do not start a worker or speculate.
    ae(traces[1000:1100], arr[1000:1100])
    _wait_prefetches(traces)
    assert slow.calls == [900]
    assert traces._blocks._worker is None

    # Navigation explicitly hints the next window, through a shared channel view.
    ae(traces[1050:1150], arr[1050:1150])
    traces[:, [1, 2]].prefetch(1200)
    _wait_prefetches(traces)
    assert sorted(traces._cache) == [3, 4]
    ae(traces[950:1050], arr[950:1050])
    traces.prefetch(850)
    _wait_prefetches(traces)
    assert sorted(traces._cache) == [2, 3, 4]
    # Block 4 comes from the cache, and the read continues to the right.
    ae(traces[1200:1300], arr[1200:1300])
    traces.prefetch(1500)
    _wait_prefetches(traces)
    assert slow.calls == [900, 1200, 600, 1500]
    traces.close()


def test_spikeinterface_reader_prefetch_failure(si_recording):
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.3)
    traces.recording = _SlowRecording(recording, fail={1200})
    traces[1000:1100]
    traces.prefetch(1200)  # block 4 fails
    _wait_prefetches(traces)
    assert 4 not in traces._cache
    ae(traces[1200:1300], arr[1200:1300])
    traces.close()


def test_spikeinterface_reader_bounded_prefetch(si_recording):
    import threading
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.1, cache_size=1)
    gate = {100: threading.Event()}
    traces.recording = slow = _SlowRecording(recording, gate=gate)
    ae(traces[:100], arr[:100])
    slow.started.clear()
    traces.prefetch(100)
    assert slow.started.wait(5)
    try:
        # Repeated forward hints and a direction change replace a single desired block.
        for start in range(300, 2000, 200):
            traces.prefetch(start)
            assert traces._blocks._active == 1
            assert traces._blocks._hint == start // 100
        traces.prefetch(0)
        assert traces._blocks._hint is None  # already cached
        traces.prefetch(200)
    finally:
        gate[100].set()
    _wait_prefetches(traces)
    ae(traces[200:300], arr[200:300])
    assert slow.calls == [0, 100, 200]
    assert list(traces._cache) == [2]
    traces.close()


def test_spikeinterface_reader_snippets(si_recording):
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.3)
    traces.recording = slow = _SlowRecording(recording)
    for start in [1000, 1100] * 10:
        ae(traces[start:start + 82], arr[start:start + 82])
    assert slow.calls == [900]
    for start in [0, 2200, 400, 1600]:
        ae(traces[start:start + 82], arr[start:start + 82])
    assert slow.calls == [900, 0, 2100, 300, 1500]
    assert traces._blocks._worker is None
    assert traces._blocks._hint is None


def test_spikeinterface_reader_cancel_queued(si_recording):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    recording, _ = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    release = threading.Event()
    traces.recording = slow = _SlowRecording(recording, gate={0: release})
    with ThreadPoolExecutor(max_workers=1) as pool:
        demand = pool.submit(lambda: traces[:100])
        assert slow.started.wait(5)
        try:
            traces.prefetch(100)
            traces.prefetch(200)
            assert traces._blocks._hint == 2
            traces.cancel_prefetch()
            assert traces._blocks._hint is None
        finally:
            release.set()
        demand.result(5)
    _wait_prefetches(traces)
    assert slow.calls == [0]
    traces.close()


@mark.parametrize('cancel', [True, False])
def test_spikeinterface_reader_replace_running_hint(si_recording, cancel):
    import threading
    recording, _ = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    release = threading.Event()
    traces.recording = slow = _SlowRecording(recording, gate={100: release})
    traces.prefetch(np.int64(100))
    assert slow.started.wait(5)
    try:
        traces.prefetch(200)
        # Returning to the running block should reuse it, without a second computation.
        traces.prefetch(150)
        traces.prefetch(100)
        if cancel:
            traces.cancel_prefetch()
    finally:
        release.set()
    _wait_prefetches(traces)
    assert slow.calls == [100]
    assert list(traces._cache) == ([] if cancel else [1])
    traces.close()


def test_spikeinterface_reader_hint_bounds(si_recording):
    recording, _ = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    for sample in [-1, traces.n_samples, traces.n_samples + 100]:
        traces.prefetch(sample)
    with raises(TypeError):
        traces.prefetch(1.5)
    assert traces._blocks._worker is None
    assert traces._blocks._hint is None
    # The last, partial block is still a valid hint.
    traces.prefetch(traces.n_samples - 1)
    _wait_prefetches(traces)
    assert list(traces._cache) == [23]
    assert traces._cache[23].shape[0] == 45
    traces.close()


@mark.parametrize('fail', [True, False])
def test_spikeinterface_reader_demand_reuses_running(si_recording, fail):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    release = threading.Event()
    traces.recording = slow = _SlowRecording(
        recording, gate={100: release}, fail={100} if fail else ())
    traces.prefetch(100)
    assert slow.started.wait(5)
    traces.prefetch(200)  # Navigation no longer needs the running block, but demand does.
    blocks = traces._blocks
    with ThreadPoolExecutor(max_workers=3) as pool:
        reads = [pool.submit(lambda: traces[100:182]) for _ in range(3)]
        try:
            with blocks._condition:
                assert blocks._condition.wait_for(
                    lambda: blocks._demands.get(1) == 3, timeout=5)
        finally:
            release.set()
        for read in reads:
            ae(read.result(5), arr[100:182])
    _wait_prefetches(traces)
    assert slow.calls == ([100, 100, 200] if fail else [100, 200])
    traces.close()


def test_spikeinterface_reader_demand_failure(si_recording):
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    traces.recording = slow = _SlowRecording(recording, fail={100})
    with raises(OSError, match='transient'):
        traces[100:200]
    ae(traces[200:300], arr[200:300])
    ae(traces[100:200], arr[100:200])
    assert slow.calls == [100, 200, 100]
    assert traces._blocks._worker is None


def test_spikeinterface_reader_serial_backend(si_recording):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    release = threading.Event()
    traces.recording = slow = _SlowRecording(recording, gate={100: release})
    traces.prefetch(100)
    assert slow.started.wait(5)
    traces.prefetch(200)

    def demand():
        return traces[500:600]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(demand)
        try:
            blocks = traces._blocks
            with blocks._condition:
                assert blocks._condition.wait_for(lambda: 5 in blocks._demands, timeout=5)
                assert slow.calls == [100]
        finally:
            release.set()
        ae(future.result(5), arr[500:600])
    _wait_prefetches(traces)
    assert slow.calls == [100, 500, 200]
    traces.close()


def test_spikeinterface_reader_close_running(si_recording):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    recording, arr = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    release = threading.Event()
    traces.recording = slow = _SlowRecording(recording, gate={100: release})
    traces.prefetch(100)
    assert slow.started.wait(5)
    traces.prefetch(200)
    worker = traces._blocks._worker
    with ThreadPoolExecutor(max_workers=1) as pool:
        demand = pool.submit(lambda: traces[500:600])
        try:
            blocks = traces._blocks
            with blocks._condition:
                assert blocks._condition.wait_for(lambda: 5 in blocks._demands, timeout=5)
            traces[:, [1, 2]].close()
            traces.close()  # idempotent across clones
            # Closing wakes waiting reads without waiting for the running backend call.
            with raises(RuntimeError, match='closed'):
                demand.result(5)
        finally:
            release.set()
            worker.join(5)
    assert not worker.is_alive()
    assert not traces._cache
    assert traces._blocks._hint is None
    assert slow.calls == [100]
    traces.prefetch(300)
    with raises(RuntimeError, match='closed'):
        traces[:10]
    assert slow.calls == [100]
    assert not traces._cache


def test_spikeinterface_reader_close_completing(si_recording):
    import threading
    recording, _ = si_recording
    traces = get_ephys_reader(recording, block_duration=.1)
    entered, release = threading.Event(), threading.Event()
    store = traces._blocks._store

    def gated_store(i, block):
        entered.set()
        assert release.wait(5)
        store(i, block)

    traces._blocks._store = gated_store
    # Hold the backend until prefetch() has returned, then close as storage completes.
    backend = threading.Event()
    traces.recording = slow = _SlowRecording(recording, gate={100: backend})
    traces.prefetch(100)
    assert slow.started.wait(5)
    backend.set()
    assert entered.wait(5)
    closer = threading.Thread(target=traces.close)
    closer.start()
    release.set()
    closer.join(5)
    traces._blocks._worker.join(5)
    assert not closer.is_alive()
    assert not traces._cache
    assert traces._blocks._hint is None
