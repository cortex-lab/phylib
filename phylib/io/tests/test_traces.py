# -*- coding: utf-8 -*-

"""Testing the EphysTraces class."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from numpy.testing import assert_equal as ae
import dask.array as da
import mtscomp

from ..traces import get_ephys_traces, EphysTraces, random_ephys_traces, extract_waveforms

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_ephys_traces_1():
    data = np.random.randn(1000, 10)
    traces = get_ephys_traces(data, 100)

    assert isinstance(traces, EphysTraces)
    assert isinstance(traces, da.Array)
    assert bool(np.all(data == traces).compute()) is True

    assert traces.dtype == data.dtype
    assert traces.shape == data.shape
    assert traces.chunks == ((100,) * 10, (10,))

    waveforms = extract_waveforms(traces, [5, 50, 100, 900], [1, 4, 7], n_samples_waveforms=10)
    assert waveforms.shape == (4, 10, 3)

    traces_sub = traces.subset_time_range(2.5, 7.5)
    assert traces_sub.shape == (500, 10)
    assert bool(np.all(traces[250:750, :] == traces_sub).compute()) is True


def test_ephys_traces_2(tempdir):
    data = (50 * np.random.randn(1000, 10)).astype(np.int16)
    sample_rate = 100
    path = tempdir / 'data.bin'

    with open(path, 'wb') as f:
        data.tofile(f)

    out = path.parent / 'data.cbin'
    outmeta = path.parent / 'data.ch'
    mtscomp.compress(
        path, out, outmeta, sample_rate=sample_rate,
        n_channels=data.shape[1], dtype=data.dtype,
        n_threads=1, check_after_compress=False, quiet=True)
    reader = mtscomp.decompress(out, outmeta, check_after_decompress=False, quiet=True)

    for obj in (reader, out, path):
        traces = get_ephys_traces(
            obj, n_channels_dat=data.shape[1], sample_rate=sample_rate, dtype=data.dtype)

        assert isinstance(traces, EphysTraces)
        assert isinstance(traces, da.Array)

        assert traces.dtype == data.dtype
        assert traces.shape == data.shape
        assert traces.chunks == ((100,) * 10, (10,))
        assert traces.duration == 10.0

        assert bool(np.all(data == traces).compute()) is True
        assert traces.chunk_bounds == reader.chunk_bounds

        spike_samples = np.array([5, 50, 100, 901])
        spike_chunks = traces._get_time_chunks(spike_samples / 100.)
        ae(spike_chunks, [0, 0, 1, 9])

        waveforms = extract_waveforms(traces, spike_samples, [1, 4, 7], 10)
        assert waveforms.shape == (4, 10, 3)

        traces_sub = traces.subset_time_range(2.5, 7.5)
        assert traces_sub.shape == (500, 10)
        assert bool(np.all(traces[250:750, :] == traces_sub).compute()) is True

        assert list((i0, i1) for (i0, i1, _) in traces.iter_chunks()) == list(
            zip(range(0, 1000, 100), range(100, 1001, 100)))
        ae(traces.get_chunk(-1), data[:100, :])
        ae(traces.get_chunk(0), data[:100, :])
        ae(traces.get_chunk(8), data[800:900, :])
        ae(traces.get_chunk(9), data[900:, :])
        ae(traces.get_chunk(10), data[900:, :])


def test_ephys_traces_3(tempdir):
    data = (50 * np.random.randn(1001, 10)).astype(np.int16)
    sample_rate = 100
    path = tempdir / 'data.bin'

    with open(path, 'wb') as f:
        data.tofile(f)

    traces = get_ephys_traces(path, sample_rate=sample_rate, dtype=np.int16, n_channels_dat=10)

    assert isinstance(traces, EphysTraces)

    assert traces.dtype == data.dtype
    assert traces.shape == data.shape
    assert traces.chunks == ((100,) * 10 + (1,), (10,))

    assert da.all(traces == data).compute()

    assert da.all(get_ephys_traces(traces) == traces).compute()


def test_ephys_traces_4(tempdir):
    data = (50 * np.random.randn(1000, 10)).astype(np.int16)
    sample_rate = 100
    path = tempdir / 'data.npy'

    np.save(path, data)

    traces = get_ephys_traces(path, sample_rate=sample_rate)

    assert isinstance(traces, EphysTraces)

    assert traces.dtype == data.dtype
    assert traces.shape == data.shape
    assert traces.chunks == ((100,) * 10, (10,))

    assert da.all(traces == data).compute()


def test_ephys_traces_5(tempdir):
    data = (50 * np.random.randn(1000, 10)).astype(np.int16)
    sample_rate = 100
    path = tempdir / 'data.bin'

    with open(path, 'wb') as f:
        data.tofile(f)

    traces = get_ephys_traces(
        (path, path), sample_rate=sample_rate, dtype=np.int16, n_channels_dat=10)

    assert isinstance(traces, EphysTraces)

    assert traces.dtype == data.dtype
    assert traces.shape == (2000, 10)
    assert len(traces.chunks[0]) == 20


def test_random_ephys_traces():
    assert random_ephys_traces(1000, 12, sample_rate=100).shape == (1000, 12)
