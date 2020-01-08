# -*- coding: utf-8 -*-

"""EphysTraces class."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
import dask.array as da
import mtscomp

from .array import _clip

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _extract_waveforms(traces, sample, channel_ids=None, n_samples_waveforms=None):
    nsw = n_samples_waveforms
    dur = traces.shape[0]
    a = nsw // 2
    b = nsw - a
    assert traces.ndim == 2
    assert nsw > 0
    assert a + b == nsw
    if channel_ids is None:  # pragma: no cover
        channel_ids = slice(None, None, None)
    t0, t1 = int(sample - a), int(sample + b)
    t0, t1 = _clip(t0, 0, dur), _clip(t1, 0, dur)
    # Extract the waveforms.
    w = traces[t0:t1][:, channel_ids]
    # Pad with zeros.
    bef = aft = 0
    if t0 == 0:  # pragma: no cover
        bef = nsw - w.shape[0]
    if t1 == dur:  # pragma: no cover
        aft = nsw - w.shape[0]
    assert bef + w.shape[0] + aft == nsw
    if bef > 0 or aft > 0:  # pragma: no cover
        w = np.pad(w, ((bef, aft), (0, 0)), 'constant')
    assert w.shape[0] == nsw
    return w


#------------------------------------------------------------------------------
# EphysTraces
#------------------------------------------------------------------------------

class EphysTraces(da.Array):
    def __new__(cls, *args, **kwargs):
        format = kwargs.pop('format', 'flat')
        sample_rate = kwargs.pop('sample_rate', None)
        assert sample_rate > 0
        self = super(EphysTraces, cls).__new__(cls, *args, **kwargs)
        self.format = format
        self.sample_rate = sample_rate
        self.duration = self.shape[0] / self.sample_rate
        return self

    @property
    def chunk_bounds(self):
        return [0] + np.cumsum([self.chunks[0]]).tolist()

    def _get_time_chunks(self, spike_times):
        """Return the time chunk indices of every spike."""
        return np.searchsorted(self.chunk_bounds, spike_times, side='right') - 1

    def extract_waveforms(self, spike_times, channel_ids, n_samples_waveforms=None):
        """Extract waveforms for a given set of spikes, on certain channels."""
        # Create the output array.
        ns = len(spike_times)
        nsw = n_samples_waveforms
        assert nsw > 0, "Please specify n_samples_waveforms > 0"
        nc = len(channel_ids)
        out = np.zeros((ns, nsw, nc), dtype=np.float64)

        # NOTE: put those chunks in the mtscomp cache in parallel.
        if self.format == 'mtscomp':
            # Find the time chunks where the spikes belong.
            chunks = np.unique(self._get_time_chunks(spike_times))
            pool = self.reader.start_thread_pool()
            self.reader.decompress_chunks(chunks, pool=pool)
            self.reader.stop_thread_pool()

        # Extract the spike waveforms.
        for i, ts in enumerate(spike_times):
            # Extract waveforms on the fly from raw data.
            out[i, ...] = _extract_waveforms(
                self, ts, channel_ids=channel_ids, n_samples_waveforms=nsw)

        return out

    def subset_time_range(self, t0, t1):
        """Return a new EphysTraces instance for a subset of the time range."""
        i0, i1 = int(round(t0 * self.sample_rate)), int(round(t1 * self.sample_rate))
        return from_dask(self[i0:i1, :], sample_rate=self.sample_rate, format=self.format)


def from_dask(arr, **kwargs):
    """From dask.array.Array instance to EphysTraces instance."""
    return EphysTraces(arr.dask, arr.name, arr.chunks, dtype=arr.dtype, shape=arr.shape, **kwargs)


def from_mtscomp(reader):
    """From a mtscomp-compressed array to EphysTraces."""
    name = 'mtscomp_traces'
    # Use the compression chunks as dask chunks.
    chunks = (tuple(np.diff(reader.chunk_bounds)), (reader.n_channels,))

    def f(i):
        return reader._decompress_chunk(i)[1]
    dask = {(name, i, 0): (f, i) for i in range(reader.n_chunks)}
    assert len(chunks[0]) == len(dask)
    out = EphysTraces(
        dask, name, chunks, dtype=reader.dtype, shape=reader.shape,
        format='mtscomp', sample_rate=reader.sample_rate)
    out.reader = reader
    return out


def from_array(arr, sample_rate):
    """From any NumPy-like array to EphysTraces instance."""
    n_samples, n_channels = arr.shape
    chunk_shape = (int(sample_rate), n_channels)
    dask_arr = da.from_array(arr, chunk_shape, name='traces')
    return from_dask(dask_arr, sample_rate=sample_rate)


def create_traces(obj, sample_rate=None):
    """Get an EphysTraces instance."""
    if isinstance(obj, mtscomp.Reader):
        return from_mtscomp(obj)
    else:
        assert sample_rate, "Please specify a sample rate."
        assert sample_rate > 0, "Please specify a sample rate as a strictly positive number."
        return from_array(obj, sample_rate)
