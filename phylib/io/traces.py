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

def _extract_waveform(traces, sample, channel_ids=None, n_samples_waveforms=None):
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
    w = traces[t0:t1, channel_ids]
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


def extract_waveforms(traces, spike_samples, channel_ids, n_samples_waveforms=None):
    """Extract waveforms for a given set of spikes, on certain channels."""
    # Create the output array.
    ns = len(spike_samples)
    nsw = n_samples_waveforms
    assert nsw > 0, "Please specify n_samples_waveforms > 0"
    nc = len(channel_ids)
    out = np.zeros((ns, nsw, nc), dtype=np.float64)
    # Extract the spike waveforms.
    out = da.concatenate((_extract_waveform(
        traces, ts, channel_ids=channel_ids, n_samples_waveforms=nsw)[np.newaxis, ...]
        for i, ts in enumerate(spike_samples)), axis=0)
    out -= da.median(out, axis=1)[:, np.newaxis, :]
    return out.compute()


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

    - reader: BaseEphysReader instance
    - sample_rate: float
    - duration: float
    - chunk_bounds : array
    - n_chunks : int

    Methods
    -------

    - subset_time_range(t0, t1): return a view of the object, with a smaller time interval
    - iter_chunks(): yield tuples (i0, i1, chunk)
    - get_chunk(chunk_idx)

    """

    def __new__(cls, *args, **kwargs):
        reader = kwargs.pop('reader')
        # sample_rate = kwargs.pop('sample_rate', None) or 0
        sample_rate = reader.sample_rate
        assert sample_rate > 0

        self = super(EphysTraces, cls).__new__(cls, *args, **kwargs)

        self.reader = reader
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
        return np.searchsorted(
            self.chunk_bounds, np.asarray(spike_times) * self.sample_rate, side='right') - 1

    def subset_time_range(self, t0, t1):
        """Return a new EphysTraces instance for a subset of the time range."""
        i0, i1 = int(round(t0 * self.sample_rate)), int(round(t1 * self.sample_rate))
        return from_dask(self[i0:i1, :], reader=self.reader)

    def __getitem__(self, item):
        out = super(EphysTraces, self).__getitem__(item)
        return from_dask(out, reader=self.reader)


#------------------------------------------------------------------------------
# Readers
#------------------------------------------------------------------------------

def from_dask(arr, **kwargs):
    """From dask.array.Array instance to EphysTraces instance."""
    return EphysTraces(arr.dask, arr.name, arr.chunks, dtype=arr.dtype, shape=arr.shape, **kwargs)


class BaseEphysReader(object):
    format = ''
    sample_rate = 0

    def get_traces(self):
        raise NotImplementedError()


class ChunkEphysReader(BaseEphysReader):
    n_channels = 0
    chunk_bounds = ()
    dtype = None

    def _set_chunk_bounds(self, n_samples):
        assert self.sample_rate
        self.chunk_bounds = tuple(np.arange(0, n_samples + 1, int(self.sample_rate)))
        if self.chunk_bounds[-1] != n_samples:
            self.chunk_bounds += (n_samples,)

    def _load_chunk(self, chunk_idx):
        raise NotImplementedError()

    def get_traces(self):
        name = 'traces'
        chunks = (tuple(np.diff(self.chunk_bounds)), (self.n_channels,))
        n_chunks = len(chunks[0])
        n_samples = self.chunk_bounds[-1]
        shape = (n_samples, self.n_channels)
        dask = {(name, i, 0): (self._load_chunk, i) for i in range(n_chunks)}
        assert len(chunks[0]) == len(dask)
        return EphysTraces(
            dask, name, chunks, dtype=self.dtype, shape=shape, reader=self)


class FlatEphysReader(ChunkEphysReader):
    format = 'flat'

    def __init__(self, path, n_channels=None, dtype=None, offset=None, sample_rate=None):
        self.path = Path(path)
        self.n_channels = n_channels
        self.dtype = np.dtype(self.dtype if self.dtype is not None else np.int16)
        self.offset = offset or 0
        self.sample_rate = sample_rate
        assert self.sample_rate > 0
        # Find the number of samples.
        assert self.n_channels > 0
        fsize = path.stat().st_size
        item_size = np.dtype(dtype).itemsize
        n_samples = (fsize - self.offset) // (item_size * self.n_channels)
        self._set_chunk_bounds(n_samples)

    def _load_chunk(self, chunk_idx):
        i0, i1 = self.chunk_bounds[chunk_idx:chunk_idx + 2]
        count = (i1 - i0) * self.n_channels
        item_size = np.dtype(self.dtype).itemsize
        offset = i0 * self.n_channels * item_size
        shape = (i1 - i0, self.n_channels)
        return np.fromfile(self.path, dtype=self.dtype, count=count, offset=offset).reshape(shape)


class NpyEphysReader(ChunkEphysReader):
    format = 'npy'

    def __init__(self, path, sample_rate=None):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self._arr = np.load(str(self.path), mmap_mode='r')
        self.n_channels = self._arr.shape[1]
        self.dtype = np.dtype(self._arr.dtype)
        self._set_chunk_bounds(self._arr.shape[0])

    def _load_chunk(self, chunk_idx):
        i0, i1 = self.chunk_bounds[chunk_idx:chunk_idx + 2]
        return self._arr[i0:i1, :]


class MtscompEphysReader(ChunkEphysReader):
    format = 'mtscomp'

    def __init__(self, mts_reader):
        self.mts_reader = mts_reader
        self.sample_rate = mts_reader.sample_rate
        self.n_channels = mts_reader.shape[1]
        self.dtype = np.dtype(mts_reader.dtype)
        self._set_chunk_bounds(mts_reader.shape[0])

    def _load_chunk(self, chunk_idx):
        return self.mts_reader._decompress_chunk(chunk_idx)[1]


class ArrayEphysReader(BaseEphysReader):
    format = 'array'

    def __init__(self, arr, sample_rate=None):
        self.arr = arr
        self.sample_rate = sample_rate

    def get_traces(self):
        arr = self.arr
        n_samples, n_channels = arr.shape
        chunk_shape = (int(self.sample_rate or n_samples), n_channels)  # 1 second chunk
        dask_arr = da.from_array(arr, chunk_shape, name='traces')
        return from_dask(dask_arr, reader=self)


class DaskEphysReader(BaseEphysReader):
    format = 'dask'

    def __init__(self, arr, sample_rate=None):
        self.arr = arr
        self.sample_rate = sample_rate

    def get_traces(self):
        return from_dask(self.arr, reader=self)


def get_ephys_traces(obj, sample_rate=None, **kwargs):
    """Get an EphysTraces instance from any NumPy-like object of file path.

    Return None if data file(s) not available.

    """
    if 'n_channels_dat' in kwargs:
        kwargs['n_channels'] = kwargs.pop('n_channels_dat')
    if isinstance(obj, mtscomp.Reader):
        logger.debug("Loading mtscomp traces from `%s`.", obj)
        reader = MtscompEphysReader(obj)
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
            reader = mtscomp.Reader()
            reader.open(path)
            reader = MtscompEphysReader(reader)
        # flat binary file
        elif ext in ('.dat', '.bin'):
            logger.debug("Loading traces from flat file `%s`.", path)
            # return from_flat_file
            reader = FlatEphysReader(path, sample_rate=sample_rate, **kwargs)
        elif ext == '.npy':
            reader = NpyEphysReader(obj, sample_rate=sample_rate)
            # TODO: other standard binary formats
        else:  # pragma: no cover
            raise IOError("Unknown file extension: %s.", ext)
    elif isinstance(obj, (tuple, list)):
        # Concatenation along the time axis of multiple raw data files/objects.
        arrs = [get_ephys_traces(o, sample_rate=sample_rate, **kwargs) for o in obj]
        arrs = [arr for arr in arrs if arr is not None and len(arr) > 0]
        if not arrs:
            return
        out = da.concatenate(arrs, axis=0)
        out = from_dask(out, reader=arrs[0].reader)
        return out
    elif isinstance(obj, EphysTraces):
        return obj
    elif isinstance(obj, da.Array):
        logger.debug("Loading traces from dask array.")
        reader = DaskEphysReader(obj, sample_rate=sample_rate)
    else:
        logger.debug("Loading traces from array.")
        assert sample_rate, "Please specify a sample rate."
        assert sample_rate > 0, "Please specify a sample rate as a strictly positive number."
        reader = ArrayEphysReader(obj, sample_rate=sample_rate)
    assert isinstance(reader, BaseEphysReader)
    return reader.get_traces()


def random_ephys_traces(n_samples, n_channels, sample_rate=None):
    """Return random ephys traces."""
    arr = da.random.normal(size=(n_samples, n_channels))
    return get_ephys_traces(arr, sample_rate=sample_rate)
