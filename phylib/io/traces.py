# -*- coding: utf-8 -*-

"""EphysReader class."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import copy
import logging
from functools import reduce
from math import ceil
import multiprocessing as mp
from operator import mul
from pathlib import Path

import numpy as np
from numpy.lib.format import (
    _check_version, _write_array_header, dtype_to_descr)
import mtscomp
from tqdm import tqdm

from .array import _index_of

logger = logging.getLogger(__name__)


# NOTE: no real need for chunking unless with mtscomp compressed datasets
DEFAULT_CHUNK_DURATION = 600.0  # seconds


EPHYS_RAW_EXTENSIONS = ('.dat', '.bin', '.raw', '.mda')


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def prod(l):
    return reduce(mul, l, 1)


def _get_subitems(bounds, item):
    """Given a list of part/chunk bounds and an item passed to __getitem__(), return
    a list (part_or_chunk_idx, subitem) to be concatenated over the first axis.
    subitem is either a slice or a list/array of indices.

    https://github.com/cortex-lab/phylib/blob/master/phylib/io/array.py
    """
    if isinstance(item, slice):
        start, stop, step = item.start, item.stop, item.step
        start = start or bounds[0]
        stop = stop or bounds[-1]
        if start < 0:
            start = start % bounds[-1]
        start = min(start, bounds[-1])
        if stop < 0:
            stop = stop % bounds[-1]
        stop = min(stop, bounds[-1])
        step = step or 1
        # NOTE: only support step == 1 for now
        assert step == 1
        assert 0 <= start <= bounds[-1]
        assert 0 <= stop <= bounds[-1]
        first_chunk, last_chunk = _find_chunks(bounds, [start, stop - 1])
        out = []
        for chunk in range(first_chunk, last_chunk + 1):
            i0, i1 = bounds[chunk:chunk + 2]
            chunk_start = max(0, start - i0)
            chunk_stop = min(i1 - i0, stop - i0)
            assert chunk_start >= 0
            assert chunk_stop <= i1
            out.append((chunk, slice(chunk_start, chunk_stop, step)))
        return out
    elif isinstance(item, (list, np.ndarray)):
        item = np.asarray(item, dtype=np.int64)
        # NOTE: only support ordered lists for now
        if len(item) >= 2:
            assert np.all(np.diff(item))
        bounds = np.asarray(bounds)
        chunks = _find_chunks(bounds, item)
        out = []
        for chunk in np.unique(chunks):
            if chunk >= len(bounds) - 1:
                raise IndexError()
            i0, i1 = bounds[chunk:chunk + 2]
            out.append((chunk, item[(i0 <= item) & (item < i1)] - i0))
        return out
    elif isinstance(item, tuple):
        return _get_subitems(bounds, item[0])
    elif isinstance(item, (int, np.generic)):
        item = int(item)
        if item < 0:
            item = item % bounds[-1]
        chunk = _find_chunks(bounds, [item])[0]
        if chunk >= len(bounds) - 1:
            raise IndexError()
        return [(chunk, item - bounds[chunk])]


def _item_length(bounds, item):  # pragma: no cover
    """Return the size of the __getitem__() output as a function of its input."""
    total = bounds[-1] - bounds[0]
    if isinstance(item, slice):
        start = item.start or bounds[0]
        stop = item.stop or total
        if stop < 0:
            stop += total
        if stop > total:
            stop = total
        step = item.step or 1
        return ceil((stop - start) / float(step))
    elif isinstance(item, (list, np.ndarray)):
        return len(item)
    elif isinstance(item, tuple):
        return _item_length(bounds, item[0])
    elif isinstance(item, (int, np.generic)):
        return 1


def _find_chunks(bounds, arr):
    """Return the chunk index of each element in arr, given the chunk bounds."""
    return np.searchsorted(bounds, arr, 'right') - 1


def _get_part_bounds(arrs):
    """Return the part bounds for multiple NumPy-like arrays."""
    return [0] + list(np.cumsum([arr.shape[0] for arr in arrs]))


def _get_chunk_bounds(arr_sizes, chunk_size):
    """Get regular chunks from multiple concatenated NumPy-like arrays."""
    assert chunk_size > 0
    b = []
    n = 0
    for arr_size in arr_sizes:
        ch = list(range(n, n + arr_size + 1, chunk_size))
        if b and ch and ch[0] == b[-1]:
            ch = ch[1:]
        b.extend(ch)
        if b[-1] != n + arr_size:
            b.append(n + arr_size)
        n += arr_size
    return b


def _apply_op(op, arg, arr):
    if op == 'cols':
        return arr[:, arg]
    f = getattr(arr, '__%s__' % op)
    return f(arg) if arg is not None else f()


def _memmap_flat(path, dtype=None, n_channels=None, offset=0, mode='r+'):
    path = Path(path)
    # Find the number of samples.
    assert n_channels > 0
    fsize = path.stat().st_size
    item_size = np.dtype(dtype).itemsize
    if (fsize - offset) % (item_size * n_channels) != 0:
        logger.warning('Inconsistent number of channels between the '
                       'params file and the binary dat file')
    n_samples = (fsize - offset) // (item_size * n_channels)
    shape = (n_samples, n_channels)
    return np.memmap(path, dtype=dtype, offset=offset, shape=shape, mode=mode)


#------------------------------------------------------------------------------
# EphysReader
#------------------------------------------------------------------------------

class BaseEphysReader(object):
    # To be set in child classes:
    sample_rate = 0
    n_channels = 0
    chunk_bounds = ()  # [0, ..., n_samples]
    part_bounds = ()  # [0, ..., part_size_0, ..., part_size_n]
    dtype = None
    name = 'default'
    dir_path = None

    ndim = 2
    sample_onset = 0
    sample_offset = None

    def __init__(self):
        self._ops = []

    @property
    def n_chunks(self):
        return max(0, len(self.chunk_bounds) - 1)

    @property
    def n_samples(self):
        # TODO: take subset interval into account
        return self.chunk_bounds[-1]

    @property
    def n_parts(self):
        return max(0, len(self.part_bounds) - 1)

    @property
    def shape(self):
        return (self.n_samples, self.n_channels)

    @property
    def duration(self):
        return self.n_samples / float(self.sample_rate)

    def _get_part(self, part_idx, subitem):
        """Return the requested item[] of a part of the data. To be overriden."""
        raise NotImplementedError()

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 1:  # pragma: no cover
                item = item[0]
            elif len(item) == 2:
                # Lazy indexing on the second axis, e.g. for channel mapping.
                cols = item[1]
                # NOTE the self = here.
                self = self._append_op('cols', cols)
                if item[0] == slice(None, None, None):
                    # traces[:, cols] should return a cloned EphysReader instance.
                    return self
                item = item[0]
            else:
                raise NotImplementedError()
        # TODO: take interval into account
        # item = _subset_interval(interval, item)
        to_concat = []
        # Obtain the requested parts.
        for part_idx, subitem in _get_subitems(self.part_bounds, item):
            to_concat.append(self._get_part(part_idx, subitem))
        # Concatenate the parts.
        out = np.vstack(to_concat)
        return self._apply_ops(out)

    def _append_op(self, op, arg=None):
        clone = copy.copy(self)
        # NOTE: make sure the clone instance has its own ops list so as to avoid side effects.
        clone._ops = list(self._ops)
        clone._ops.append((op, arg))
        return clone

    def _apply_ops(self, arr):
        for op, arg in self._ops:
            arr = _apply_op(op, arg, arr)
        return arr

    def __add__(self, arg):
        return self._append_op('add', arg)

    def __radd__(self, arg):
        return self._append_op('radd', arg)

    def __sub__(self, arg):
        return self._append_op('sub', arg)

    def __rsub__(self, arg):
        return self._append_op('rsub', arg)

    def __mul__(self, arg):
        return self._append_op('mul', arg)

    def __rmul__(self, arg):
        return self._append_op('rmul', arg)

    def __div__(self, arg):  # pragma: no cover
        return self._append_op('div', arg)

    def __rdiv__(self, arg):  # pragma: no cover
        return self._append_op('rdiv', arg)

    def __floordiv__(self, arg):
        return self._append_op('floordiv', arg)

    def __truediv__(self, arg):
        return self._append_op('truediv', arg)

    def __rfloordiv__(self, arg):
        return self._append_op('rfloordiv', arg)

    def __rtruediv__(self, arg):
        return self._append_op('rtruediv', arg)

    def __pos__(self):
        return self._append_op('pos')

    def __neg__(self):
        return self._append_op('neg')

    def __pow__(self, arg):
        return self._append_op('pow', arg)

    def __rpow__(self, arg):
        return self._append_op('rpow', arg)

    def subset_time_range(self, interval):
        raise NotImplementedError()

    def iter_chunks(self, cache=True):
        for i0, i1 in zip(self.chunk_bounds[:-1], self.chunk_bounds[1:]):
            yield i0, i1


class FlatEphysReader(BaseEphysReader):
    def __init__(self, paths, sample_rate=None, dtype=None, offset=0, n_channels=None, mode='r',
                 **kwargs):
        super(FlatEphysReader, self).__init__()
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self._paths = [Path(p) for p in paths]
        assert all(p.exists() for p in self._paths)
        self.name = paths[0].stem
        self.dir_path = paths[0].parent
        self._mmaps = [
            _memmap_flat(path, dtype=dtype, n_channels=n_channels, offset=offset, mode=mode)
            for path in paths]

        self.sample_rate = sample_rate
        self.dtype = dtype
        self.n_channels = n_channels
        chunk_size = int(round(DEFAULT_CHUNK_DURATION * sample_rate))
        self.part_bounds = _get_part_bounds(self._mmaps)
        self.chunk_bounds = _get_chunk_bounds(
            [arr.shape[0] for arr in self._mmaps], chunk_size=chunk_size)

        assert self.sample_rate > 0
        assert self.n_channels >= 0

    def _get_part(self, part_idx, subitem):
        """To be overriden."""
        return self._mmaps[part_idx][subitem]


class MtscompEphysReader(BaseEphysReader):
    def __init__(self, reader, **kwargs):
        super(MtscompEphysReader, self).__init__()
        if isinstance(reader, (tuple, list)):  # pragma: no cover
            assert reader
            # TODO: support concatenation of multiple .cbin files. Currently, only take the first.
            if len(reader) >= 2:
                logger.warning(
                    "Support of multiple concatenate .cbin files is not supported yet. "
                    "Taking the first file only.")
            reader = reader[0]
        assert isinstance(reader, mtscomp.Reader)
        self.reader = reader
        self.name = reader.cdata.name
        self.dir_path = Path(self.name).parent
        self.sample_rate = reader.sample_rate
        self.dtype = reader.dtype
        self.n_channels = reader.n_channels
        self.part_bounds = [0, reader.n_samples]  # TODO: support multiple concatenated readers
        self.chunk_bounds = reader.chunk_bounds

    def _get_part(self, part_idx, subitem):
        assert part_idx == 0
        return self.reader[subitem]

    def iter_chunks(self, cache=True):
        """Iterate over multiple chunks that are decompressed in parallel."""
        reader = self.reader

        if cache:
            # Create the thread pool.
            reader.start_thread_pool()
            # Make sure all chunks from a batch are cached.
            reader.set_cache_size(reader.n_batches + 2)

        for batch in range(reader.n_batches):
            first_chunk = reader.batch_size * batch  # first included
            last_chunk = min(reader.batch_size * (batch + 1), reader.n_chunks)  # last excluded
            assert 0 <= first_chunk < last_chunk <= reader.n_chunks

            if cache:
                logger.debug(
                    "Processing batch #%d/%d with chunks %s.",
                    batch + 1,
                    reader.n_batches, ', '.join(map(str, range(first_chunk, last_chunk))))
                # Decompress all chunks in the batch.
                reader.decompress_chunks(range(first_chunk, last_chunk), reader.pool)

            # Do not include the last chunk so as to cache the next chunk (useful when extracting
            # waveforms).
            first_chunk = max(first_chunk - 1, 0)
            last_chunk = max(first_chunk, last_chunk - 1)
            yield reader.chunk_bounds[first_chunk], reader.chunk_bounds[last_chunk]
        # Last chunk.
        yield reader.chunk_bounds[last_chunk], reader.chunk_bounds[last_chunk + 1]

        # Close the thread pool.
        if cache:
            reader.stop_thread_pool()


class ArrayEphysReader(BaseEphysReader):
    def __init__(self, arr, **kwargs):
        super(ArrayEphysReader, self).__init__()
        self._arr = arr
        self.sample_rate = kwargs.pop('sample_rate', None)
        assert self.sample_rate > 0
        self.dtype = arr.dtype
        self.n_channels = arr.shape[1]
        self.part_bounds = [0, arr.shape[0]]
        chunk_size = int(round(DEFAULT_CHUNK_DURATION * self.sample_rate))
        self.chunk_bounds = _get_chunk_bounds([arr.shape[0]], chunk_size=chunk_size)

    def _get_part(self, part_idx, subitem):
        assert part_idx == 0
        return self._arr[subitem]


class NpyEphysReader(ArrayEphysReader):
    def __init__(self, path, **kwargs):
        if isinstance(path, (tuple, list)):
            if len(path) != 1:
                raise ValueError("There should be exactly one path to a npy file.")
            path = path[0]
        path = Path(path)
        self.name = path.stem
        self.dir_path = path.parent
        self._arr = np.load(path, mmap_mode='r')  # TODO: support for multiple npy files
        super(NpyEphysReader, self).__init__(self._arr, **kwargs)


class RandomEphysReader(BaseEphysReader):
    name = 'random'

    def __init__(self, n_samples, n_channels, sample_rate=None, **kwargs):
        super(RandomEphysReader, self).__init__()
        self.sample_rate = sample_rate
        assert self.sample_rate > 0
        self.dtype = np.float32
        self.n_channels = n_channels
        self.part_bounds = [0, n_samples]
        chunk_size = int(round(DEFAULT_CHUNK_DURATION * self.sample_rate))
        self.chunk_bounds = _get_chunk_bounds([n_samples], chunk_size=chunk_size)

    def _get_part(self, part_idx, subitem):
        assert part_idx == 0
        n = _item_length(self.chunk_bounds, subitem)
        return np.random.randn(n, self.n_channels).astype(np.float32)


#------------------------------------------------------------------------------
# High-level functions
#------------------------------------------------------------------------------

def _get_ephys_constructor(obj, **kwargs):
    """Return the class, argument, and kwargs to create an Ephys instance from any
    compatible Python object."""
    if 'n_channels_dat' in kwargs:  # pragma: no cover
        kwargs['n_channels'] = kwargs.pop('n_channels_dat')
    if isinstance(obj, mtscomp.Reader):
        return (MtscompEphysReader, obj, kwargs)
    elif isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.exists():  # pragma: no cover
            logger.warning("File %s does not exist.", path)
            return None, None, {}
        assert path.exists()
        ext = path.suffix
        assert ext, "No extension found in file `%s`" % path
        # Mtscomp file
        if ext == '.cbin':
            reader = mtscomp.Reader(n_threads=mp.cpu_count() // 2)
            reader.open(path)
            return (MtscompEphysReader, reader, kwargs)
        # Flat binary file
        elif ext in EPHYS_RAW_EXTENSIONS:
            return (FlatEphysReader, path, kwargs)
        elif ext == '.npy':
            return (NpyEphysReader, obj, kwargs)
            # TODO: other standard binary formats
        else:  # pragma: no cover
            raise IOError("Unknown file extension: `%s`." % ext)
    elif isinstance(obj, (tuple, list)):
        if obj:
            # Concatenate the main argument to the constructor.
            klass, arg, kwargs = _get_ephys_constructor(obj[0], **kwargs)
            arg = [_get_ephys_constructor(o)[1] for o in obj]
            return (klass, arg, kwargs)
    else:
        return (ArrayEphysReader, obj, kwargs)


def get_ephys_reader(obj, **kwargs):
    """Get an EphysReader instance from any NumPy-like object of file path.

    Return None if data file(s) not available.

    """
    klass, arg, kwargs = _get_ephys_constructor(obj, **kwargs)
    if not klass:
        return
    return klass(arg, **kwargs)


#------------------------------------------------------------------------------
# Waveform extractor
#------------------------------------------------------------------------------

def get_spike_waveforms(spike_ids, channel_ids, spike_waveforms=None, n_samples_waveforms=None):
    """Get spike waveforms from precomputed doubly sparse spike waveforms array.

    The `spike_waveforms` object is a Bunch with attributes
    `spike_ids`, `spike_channels`, `waveforms`.

    """
    assert spike_waveforms
    # Make sure the requested spikes all belong to the spike_waveforms object.
    assert np.all(np.isin(spike_ids, spike_waveforms.spike_ids))
    spike_ids_rel = _index_of(spike_ids, spike_waveforms.spike_ids)
    ns = len(spike_ids)
    nsw = n_samples_waveforms
    assert nsw > 0
    nc = len(channel_ids)
    assert nc > 0
    out = np.zeros((ns, nsw, nc), dtype=spike_waveforms.waveforms.dtype)
    # Extract the spike waveforms.
    for i, sid in enumerate(spike_ids_rel):
        ind = spike_waveforms.spike_channels[sid, :]
        channel_common = np.intersect1d(channel_ids, ind)
        if len(channel_ids) > 0:
            cols0 = _index_of(channel_common, channel_ids)
            cols1 = _index_of(channel_common, ind)
            assert len(cols0) == len(cols1)
            out[i, :, cols0] = spike_waveforms.waveforms[sid, :, cols1]
    return out


def _npy_header(shape, dtype, order='C'):  # pragma: no cover
    d = {'shape': shape}
    if order == 'C':
        d['fortran_order'] = False
    elif order == 'F':
        d['fortran_order'] = True
    else:
        # Totally non-contiguous data. We will have to make it C-contiguous
        # before writing. Note that we need to test for C_CONTIGUOUS first
        # because a 1-D array is both C_CONTIGUOUS and F_CONTIGUOUS.
        d['fortran_order'] = False
    d['descr'] = dtype_to_descr(dtype)
    return d


class NpyWriter(object):
    def __init__(self, path, shape, dtype, axis=0):
        assert axis == 0  # only concatenation along the first axis is supported right now
        # Only C order is supported at the moment.
        self.shape = shape
        self.dtype = np.dtype(dtype)
        header = _npy_header(self.shape, self.dtype)
        version = None
        _check_version(version)
        self.fp = open(path, 'wb')
        _write_array_header(self.fp, header, version)

    def append(self, chunk):
        if chunk.ndim == len(self.shape):
            assert chunk.shape[1:] == self.shape[1:]
        else:  # pragma: no cover
            assert chunk.shape == self.shape[1:]
        self.fp.write(chunk.tobytes())

    def close(self):
        self.fp.close()


def _extract_waveform(traces, sample, channel_ids=None, n_samples_waveforms=None):
    """Extract a single spike waveform."""
    nsw = n_samples_waveforms
    assert traces.ndim == 2
    dur = traces.shape[0]
    a = nsw // 2
    b = nsw - a
    assert nsw > 0
    assert a + b == nsw
    if channel_ids is None:  # pragma: no cover
        channel_ids = slice(None, None, None)
        n_channels = traces.shape[1]
    else:
        n_channels = len(channel_ids)
    t0, t1 = int(sample - a), int(sample + b)
    # Extract the waveforms.
    w = traces[max(0, t0):t1][:, channel_ids]
    if not isinstance(channel_ids, slice):
        w[:, channel_ids == -1] = 0
    # Deal with side effects.
    if t0 < 0:
        w = np.vstack((np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype), w))
    if t1 > dur:
        w = np.vstack((w, np.zeros((nsw - w.shape[0], n_channels), dtype=w.dtype)))
    assert w.shape == (nsw, n_channels)
    return w


def extract_waveforms(traces, spike_samples, channel_ids, n_samples_waveforms=None):
    """Extract waveforms for a given set of spikes, on certain channels."""
    # Create the output array.
    ns = len(spike_samples)
    nsw = n_samples_waveforms
    assert nsw > 0, "Please specify n_samples_waveforms > 0"
    nc = len(channel_ids)
    # Extract the spike waveforms.
    out = np.zeros((ns, nsw, nc), dtype=traces.dtype)
    for i, ts in enumerate(spike_samples):
        out[i] = _extract_waveform(
            traces, ts, channel_ids=channel_ids, n_samples_waveforms=nsw)[np.newaxis, ...]
    return out


def iter_waveforms(traces, spike_samples, spike_channels, n_samples_waveforms=None, cache=False):
    """Iterate over trace chunks and yield batches of spike waveforms."""
    spike_samples = np.asarray(spike_samples)
    spike_channels = np.asarray(spike_channels)

    assert spike_channels.ndim == 2
    assert spike_samples.shape[0] == spike_channels.shape[0]

    n_samples_waveforms = n_samples_waveforms
    n_channels_loc = spike_channels.shape[1]

    pb = tqdm(desc="Extracting waveforms", total=traces.duration)
    for i0, i1 in traces.iter_chunks(cache=cache):
        # Get spikes in chunk.
        ind = _find_chunks([i0, i1], spike_samples) == 0
        ss = spike_samples[ind]
        sc = spike_channels[ind]
        ns = len(ss)
        pb.update((i1 - i0) / traces.sample_rate)
        if ns == 0:
            continue
        # Extract the spike waveforms within the chunk.
        waveforms = np.zeros((ns, n_samples_waveforms, n_channels_loc), dtype=traces.dtype)
        for i, ss in enumerate(ss):
            channel_ids = sc[i, :]
            waveforms[i, ...] = _extract_waveform(
                traces, ss, channel_ids=channel_ids,
                n_samples_waveforms=n_samples_waveforms)
        yield waveforms
    pb.close()


def export_waveforms(
        path, traces, spike_samples, spike_channels, n_samples_waveforms=None, cache=False,
        sample2unit=1):
    """Export a selection of spike waveforms to a npy file by iterating over the data on a chunk
    by chunk basis."""
    n_spikes = len(spike_samples)
    spike_channels = np.asarray(spike_channels, dtype=np.int32)
    n_channels_loc = spike_channels.shape[1]
    shape = (n_spikes, n_samples_waveforms, n_channels_loc)
    dtype = traces.dtype if sample2unit is None else float
    writer = NpyWriter(path, shape, dtype)
    size_written = 0
    for waveforms in iter_waveforms(
            traces, spike_samples, spike_channels, n_samples_waveforms=n_samples_waveforms,
            cache=cache):
        writer.append(waveforms * sample2unit)
        size_written += waveforms.size
    writer.close()
    assert prod(shape) == size_written
