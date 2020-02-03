# -*- coding: utf-8 -*-

"""EphysTraces class."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path

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
    """A class representing raw data, an arbitrarily large array (n_samples, n_channels).
    Derives from dask.array.Array, has a few convenient methods and properties.
    Can be used as a NumPy array, but one needs to call `.compute()` eventually to get
    actual values (lazy evaluation).

    Properties
    ----------

    - format: for now, either 'flat' or 'mtscomp
    - sample_rate: float
    - duration: float
    - chunk_bounds : array
    - n_chunks : int

    Methods
    -------

    - extract_waveforms(spike_times, channel_ids): with mtscomp, load chunks in parallel.
      Most efficient when the spikes are selected from a limited number of chunks.
    - subset_time_range(t0, t1): return a view of the object, with a smaller time interval
    - iter_chunks(): yield tuples (i0, i1, chunk)
    - get_chunk(chunk_idx)

    """

    def __new__(cls, *args, **kwargs):
        format = kwargs.pop('format', 'flat')
        sample_rate = kwargs.pop('sample_rate', None) or 0
        assert sample_rate > 0
        self = super(EphysTraces, cls).__new__(cls, *args, **kwargs)
        self.format = format
        self.sample_rate = sample_rate
        return self

    @property
    def chunk_bounds(self):
        """List of the chunk bounds."""
        return [0] + np.cumsum([self.chunks[0]]).tolist()

    @property
    def n_chunks(self):
        return len(self.chunk_bounds) - 1

    @property
    def duration(self):
        return self.shape[0] / float(self.sample_rate)

    def iter_chunks(self):
        """Iterate over tuples (i0, i1, chunk_data)."""
        for i0, i1 in zip(self.chunk_bounds[:-1], self.chunk_bounds[1:]):
            yield i0, i1, self[i0:i1, ...]

    def get_chunk(self, chunk_idx):
        """Return data of a given chunk."""
        chunk_idx = _clip(chunk_idx, 0, self.n_chunks - 1)
        i0, i1 = self.chunk_bounds[chunk_idx:chunk_idx + 2]
        return self[i0:i1, ...]

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
            # NOTE: make sure the cache is large enough to keep all required chunks in memory.
            # This will make the step below (for loop for waveform extraction) faster.
            self.reader.set_cache_size(len(chunks))
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

    def __getitem__(self, item):
        out = super(EphysTraces, self).__getitem__(item)
        return from_dask(out, format=self.format, sample_rate=self.sample_rate)


def from_dask(arr, **kwargs):
    """From dask.array.Array instance to EphysTraces instance."""
    return EphysTraces(arr.dask, arr.name, arr.chunks, dtype=arr.dtype, shape=arr.shape, **kwargs)


def from_mtscomp(reader):
    """Convert an mtscomp-compressed array to an EphysTraces instance."""
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
    """Convert any NumPy-like array to an EphysTraces instance, using an 1 second chunk size."""
    n_samples, n_channels = arr.shape
    chunk_shape = (int(sample_rate or n_samples), n_channels)  # 1 second chunk
    dask_arr = da.from_array(arr, chunk_shape, name='traces')
    return from_dask(dask_arr, sample_rate=sample_rate)


def from_flat_file(path, n_channels_dat=None, dtype=None, offset=None, sample_rate=None):
    """Memmap a flat binary file and return an EphysTraces instance."""
    path = Path(path)
    # Accept mtscomp files.
    # Default dtype and offset.
    dtype = dtype if dtype is not None else np.int16
    offset = offset or 0
    # TODO: support order
    # assert order not in ('F', 'fortran')
    # Find the number of samples.
    assert n_channels_dat > 0
    fsize = path.stat().st_size
    item_size = np.dtype(dtype).itemsize
    n_samples = (fsize - offset) // (item_size * n_channels_dat)
    shape = (n_samples, n_channels_dat)
    return from_array(np.memmap(
        str(path), dtype=dtype, shape=shape, offset=offset), sample_rate)


def get_ephys_traces(obj, sample_rate=None, **kwargs):
    """Get an EphysTraces instance from any NumPy-like object of file path.

    Return None if data file(s) not available.

    """
    if isinstance(obj, mtscomp.Reader):
        logger.debug("Loading mtscomp traces from `%s`.", obj)
        return from_mtscomp(obj)
    elif isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():  # pragma: no cover
            logger.warning("File %s does not exist.", path)
            return
        assert path.exists()
        ext = path.suffix
        # mtscomp file
        if ext == '.cbin':
            logger.debug("Loading mtscomp traces from `%s`.", path)
            r = mtscomp.Reader()
            r.open(path)
            return from_mtscomp(r)
        # flat binary file
        elif ext in ('.dat', '.bin'):
            logger.debug("Loading traces from flat file `%s`.", path)
            return from_flat_file(path, sample_rate=sample_rate, **kwargs)
        elif ext == '.npy':
            return from_array(np.load(obj, mmap_mode='r'), sample_rate)
        # TODO: other standard binary formats
        else:  # pragma: no cover
            raise IOError("Unknown file extension: %s.", ext)
    elif isinstance(obj, (tuple, list)):
        # Concatenation along the time axis of multiple raw data files/objects.
        arrs = [get_ephys_traces(o, sample_rate=sample_rate, **kwargs) for o in obj]
        arrs = [arr for arr in arrs if arr is not None and len(arr) > 0]
        if not arrs:
            return
        arrs = da.concatenate(arrs, axis=0)
        return from_dask(arrs, sample_rate=sample_rate)
    else:
        logger.debug("Loading traces from array.")
        assert sample_rate, "Please specify a sample rate."
        assert sample_rate > 0, "Please specify a sample rate as a strictly positive number."
        return from_array(obj, sample_rate=sample_rate)


def random_ephys_traces(n_samples, n_channels, sample_rate=None):
    """Return random ephys traces."""
    return from_dask(da.random.normal(size=(n_samples, n_channels)), sample_rate=sample_rate)
