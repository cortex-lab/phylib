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

from ..traces import create_traces, EphysTraces

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_ephys_traces_1():
    data = np.random.randn(1000, 10)
    traces = create_traces(data, 100)

    assert isinstance(traces, EphysTraces)
    assert isinstance(traces, da.Array)
    assert bool(np.all(data == traces).compute()) is True

    assert traces.dtype == data.dtype
    assert traces.shape == data.shape
    assert traces.chunks == ((100,) * 10, (10,))

    waveforms = traces.extract_waveforms([5, 50, 100, 900], [1, 4, 7], 10)
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

    traces = create_traces(reader)

    assert isinstance(traces, EphysTraces)
    assert isinstance(traces, da.Array)

    assert traces.dtype == data.dtype
    assert traces.shape == data.shape
    assert traces.chunks == ((100,) * 10, (10,))

    assert bool(np.all(data == traces).compute()) is True
    assert traces.chunk_bounds == reader.chunk_bounds

    spike_times = [5, 50, 100, 901]
    spike_chunks = traces._get_time_chunks(spike_times)
    ae(spike_chunks, [0, 0, 1, 9])

    waveforms = traces.extract_waveforms(spike_times, [1, 4, 7], 10)
    assert waveforms.shape == (4, 10, 3)

    traces_sub = traces.subset_time_range(2.5, 7.5)
    assert traces_sub.shape == (500, 10)
    assert bool(np.all(traces[250:750, :] == traces_sub).compute()) is True
