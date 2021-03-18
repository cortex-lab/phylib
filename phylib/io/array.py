# -*- coding: utf-8 -*-

"""Utility functions for NumPy arrays."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from math import floor, ceil
from operator import itemgetter
from pathlib import Path

import numpy as np

from phylib.utils import _as_scalar, _as_scalars
from phylib.utils._types import _as_array

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

def _clip(x, a, b):
    return max(a, min(b, x))


def _range_from_slice(myslice, start=None, stop=None, step=None, length=None):
    """Convert a slice to an array of integers."""
    assert isinstance(myslice, slice)
    # Find 'step'.
    step = myslice.step if myslice.step is not None else step
    if step is None:
        step = 1
    # Find 'start'.
    start = myslice.start if myslice.start is not None else start
    if start is None:
        start = 0
    # Find 'stop' as a function of length if 'stop' is unspecified.
    stop = myslice.stop if myslice.stop is not None else stop
    if length is not None:
        stop_inferred = floor(start + step * length)
        if stop is not None and stop < stop_inferred:
            raise ValueError("'stop' ({stop}) and ".format(stop=stop) +
                             "'length' ({length}) ".format(length=length) +
                             "are not compatible.")
        stop = stop_inferred
    if stop is None and length is None:
        raise ValueError("'stop' and 'length' cannot be both unspecified.")
    myrange = np.arange(start, stop, step)
    # Check the length if it was specified.
    if length is not None:
        assert len(myrange) == length
    return myrange


def _unique(x):
    """Faster version of np.unique().

    This version is restricted to 1D arrays of non-negative integers.

    It is only faster if len(x) >> len(unique(x)).

    """
    if x is None or len(x) == 0:
        return np.array([], dtype=np.int64)
    # WARNING: only keep positive values.
    # cluster=-1 means "unclustered".
    x = _as_array(x)
    x = x[x >= 0]
    bc = np.bincount(x)
    return np.nonzero(bc)[0]


def _normalize(arr, keep_ratio=False):
    """Normalize an array into [0, 1]."""
    (x_min, y_min), (x_max, y_max) = arr.min(axis=0), arr.max(axis=0)

    if keep_ratio:
        a = 1. / max(x_max - x_min, y_max - y_min)
        ax = ay = a
        bx = .5 - .5 * a * (x_max + x_min)
        by = .5 - .5 * a * (y_max + y_min)
    else:
        ax = 1. / (x_max - x_min)
        ay = 1. / (y_max - y_min)
        bx = -x_min / (x_max - x_min)
        by = -y_min / (y_max - y_min)

    arr_n = arr.copy()
    arr_n[:, 0] *= ax
    arr_n[:, 0] += bx
    arr_n[:, 1] *= ay
    arr_n[:, 1] += by

    return arr_n


def _index_of(arr, lookup):
    """Replace scalars in an array by their indices in a lookup table.

    Implicitely assume that:

    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.

    This is not checked for performance reasons.

    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    # TODO: assertions to disable in production for performance reasons.
    # TODO: np.searchsorted(lookup, arr) is faster on small arrays with large
    # values
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


def _pad(arr, n, dir='right'):
    """Pad an array with zeros along the first axis.

    Parameters
    ----------

    n : int
        Size of the returned array in the first axis.
    dir : str
        Direction of the padding. Must be one 'left' or 'right'.

    """
    assert dir in ('left', 'right')
    if n < 0:
        raise ValueError("'n' must be positive: {0}.".format(n))
    elif n == 0:
        return np.zeros((0,) + arr.shape[1:], dtype=arr.dtype)
    n_arr = arr.shape[0]
    shape = (n,) + arr.shape[1:]
    if n_arr == n:
        assert arr.shape == shape
        return arr
    elif n_arr < n:
        out = np.zeros(shape, dtype=arr.dtype)
        if dir == 'left':
            out[-n_arr:, ...] = arr
        elif dir == 'right':
            out[:n_arr, ...] = arr
        assert out.shape == shape
        return out
    else:
        if dir == 'left':
            out = arr[-n:, ...]
        elif dir == 'right':
            out = arr[:n, ...]
        assert out.shape == shape
        return out


def _get_padded(data, start, end):
    """Return `data[start:end]` filling in with zeros outside array bounds

    Assumes that either `start<0` or `end>len(data)` but not both.

    """
    if start < 0 and end > data.shape[0]:
        raise RuntimeError()
    if start < 0:
        start_zeros = np.zeros((-start, data.shape[1]),
                               dtype=data.dtype)
        return np.vstack((start_zeros, data[:end]))
    elif end > data.shape[0]:
        end_zeros = np.zeros((end - data.shape[0], data.shape[1]),
                             dtype=data.dtype)
        return np.vstack((data[start:], end_zeros))
    else:
        return data[start:end]


def _get_data_lim(arr, n_spikes=None):
    n = arr.shape[0]
    k = max(1, n // n_spikes) if n_spikes else 1
    arr = np.abs(arr[::k])
    n = arr.shape[0]
    arr = arr.reshape((n, -1))
    return arr.max() or 1.


def get_closest_clusters(cluster_id, cluster_ids, sim_func, max_n=None):
    """Return a list of pairs `(cluster, similarity)` sorted by decreasing
    similarity to a given cluster."""
    l = [(_as_scalar(candidate), _as_scalar(sim_func(cluster_id, candidate)))
         for candidate in _as_scalars(cluster_ids)]
    l = sorted(l, key=itemgetter(1), reverse=True)
    max_n = None or len(l)
    return l[:max_n]


def _flatten(l):
    return [item for sublist in l for item in sublist]


# -----------------------------------------------------------------------------
# I/O functions
# -----------------------------------------------------------------------------

def read_array(path, mmap_mode=None):
    """Read a .npy array."""
    path = Path(path)
    file_ext = path.suffix
    if file_ext == '.npy':
        return np.load(str(path), mmap_mode=mmap_mode)
    raise NotImplementedError("The file extension `{}` is not currently supported." % file_ext)


def write_array(path, arr):
    """Write an array to a .npy file."""
    path = Path(path)
    file_ext = path.suffix
    if file_ext == '.npy':
        return np.save(str(path), arr)
    raise NotImplementedError("The file extension `{}` is not currently supported." % file_ext)


# -----------------------------------------------------------------------------
# Chunking functions
# -----------------------------------------------------------------------------

def _excerpt_step(n_samples, n_excerpts=None, excerpt_size=None):
    """Compute the step of an excerpt set as a function of the number
    of excerpts or their sizes."""
    assert n_excerpts >= 2
    step = max((n_samples - excerpt_size) // (n_excerpts - 1),
               excerpt_size)
    return step


def chunk_bounds(n_samples, chunk_size, overlap=0):
    """Return chunk bounds.

    Chunks have the form:

        [ overlap/2 | chunk_size-overlap | overlap/2 ]
        s_start   keep_start           keep_end     s_end

    Except for the first and last chunks which do not have a left/right
    overlap.

    This generator yields (s_start, s_end, keep_start, keep_end).

    """
    s_start = 0
    s_end = chunk_size
    keep_start = s_start
    keep_end = s_end - overlap // 2
    yield s_start, s_end, keep_start, keep_end

    while s_end - overlap + chunk_size < n_samples:
        s_start = s_end - overlap
        s_end = s_start + chunk_size
        keep_start = keep_end
        keep_end = s_end - overlap // 2
        if s_start < s_end:
            yield s_start, s_end, keep_start, keep_end

    s_start = s_end - overlap
    s_end = n_samples
    keep_start = keep_end
    keep_end = s_end
    if s_start < s_end:
        yield s_start, s_end, keep_start, keep_end


def excerpts(n_samples, n_excerpts=None, excerpt_size=None):
    """Yield (start, end) where start is included and end is excluded."""
    assert n_excerpts >= 2
    step = _excerpt_step(n_samples, n_excerpts=n_excerpts, excerpt_size=excerpt_size)
    for i in range(n_excerpts):
        start = i * step
        if start >= n_samples:
            break
        end = min(start + excerpt_size, n_samples)
        yield start, end


def data_chunk(data, chunk, with_overlap=False):
    """Get a data chunk."""
    assert isinstance(chunk, tuple)
    if len(chunk) == 2:
        i, j = chunk
    elif len(chunk) == 4:
        if with_overlap:
            i, j = chunk[:2]
        else:
            i, j = chunk[2:]
    else:
        raise ValueError("'chunk' should have 2 or 4 elements, not {0:d}".format(len(chunk)))
    return data[i:j, ...]


def get_excerpts(data, n_excerpts=None, excerpt_size=None):
    """Return excerpts of a data array."""
    assert n_excerpts is not None
    assert excerpt_size is not None
    if len(data) < n_excerpts * excerpt_size:
        return data
    elif n_excerpts == 0:
        return data[:0]
    elif n_excerpts == 1:
        return data[:excerpt_size]
    out = np.concatenate([
        data_chunk(data, chunk)
        for chunk in excerpts(len(data), n_excerpts=n_excerpts, excerpt_size=excerpt_size)])
    assert len(out) <= n_excerpts * excerpt_size
    return out


# -----------------------------------------------------------------------------
# Spike clusters utility functions
# -----------------------------------------------------------------------------

def _spikes_in_clusters(spike_clusters, clusters):
    """Return the ids of all spikes belonging to the specified clusters."""
    if len(spike_clusters) == 0 or len(clusters) == 0:
        return np.array([], dtype=int)
    return np.nonzero(np.in1d(spike_clusters, clusters))[0]


def _spikes_per_cluster(spike_clusters, spike_ids=None):
    """Return a dictionary {cluster: list_of_spikes}."""
    if spike_clusters is None or not len(spike_clusters):
        return {}
    if spike_ids is None:
        spike_ids = np.arange(len(spike_clusters)).astype(np.int64)
    # NOTE: this sort method is stable, so spike ids are increasing
    # among any cluster. Therefore we don't have to sort again down here,
    # when creating the spikes_in_clusters dictionary.
    rel_spikes = np.argsort(spike_clusters, kind='mergesort')
    abs_spikes = spike_ids[rel_spikes]
    spike_clusters = spike_clusters[rel_spikes]

    diff = np.empty_like(spike_clusters)
    diff[0] = 1
    diff[1:] = np.diff(spike_clusters)

    idx = np.nonzero(diff > 0)[0]
    clusters = spike_clusters[idx]

    # NOTE: we don't have to sort abs_spikes[...] here because the argsort
    # using 'mergesort' above is stable.
    spikes_in_clusters = {
        clusters[i]: abs_spikes[idx[i]:idx[i + 1]] for i in range(len(clusters) - 1)}
    spikes_in_clusters[clusters[-1]] = abs_spikes[idx[-1]:]

    return spikes_in_clusters


def _flatten_per_cluster(per_cluster):
    """Convert a dictionary {cluster: spikes} to a spikes array."""
    return np.unique(np.concatenate(list(per_cluster.values()))).astype(np.int64)


def grouped_mean(arr, spike_clusters):
    """Compute the mean of a spike-dependent quantity for every cluster.

    The two arguments should be 1D array with `n_spikes` elements.

    The output is a 1D array with `n_clusters` elements. The clusters are
    sorted in increasing order.

    """
    arr = np.asarray(arr)
    spike_clusters = np.asarray(spike_clusters)
    assert arr.shape[0] == len(spike_clusters)
    cluster_ids = _unique(spike_clusters)
    spike_clusters_rel = _index_of(spike_clusters, cluster_ids)
    spike_counts = np.bincount(spike_clusters_rel)
    assert len(spike_counts) == len(cluster_ids)
    t = np.zeros((len(cluster_ids),) + arr.shape[1:])
    # Compute the sum with possible repetitions.
    np.add.at(t, spike_clusters_rel, arr)
    return t / spike_counts.reshape((-1,) + (1,) * (arr.ndim - 1))


# -----------------------------------------------------------------------------
# Spike selection
# -----------------------------------------------------------------------------

def _times_in_chunks(times, chunks_kept):
    """Return the indices of the times that belong to a list of kept chunks."""
    ind = np.searchsorted(chunks_kept, times, side='right')
    return ind % 2 == 1


class SpikeSelector(object):
    """Select a given number of spikes per cluster among a subset of the chunks."""
    def __init__(
            self, get_spikes_per_cluster=None, spike_times=None,
            chunk_bounds=None, n_chunks_kept=None):
        self.get_spikes_per_cluster = get_spikes_per_cluster
        self.spike_times = spike_times
        self.chunks_kept = []
        n_chunks = len(chunk_bounds) - 1

        for i in range(0, n_chunks, max(1, int(ceil(n_chunks / n_chunks_kept)))):
            self.chunks_kept.extend(chunk_bounds[i:i + 2])
        self.chunks_kept = np.array(self.chunks_kept)

    def __call__(self, n_spk_clu, cluster_ids, subset_chunks=False, subset_spikes=None):
        """Select about n_spk_clu random spikes from each of the requested clusters, only
        in the kept chunks."""
        if not len(cluster_ids):
            return np.array([], dtype=np.int64)
        # Start with all spikes from each cluster.
        selection = {}
        for cluster in cluster_ids:
            # Get all spikes from that cluster.
            spike_ids = self.get_spikes_per_cluster(cluster)
            # Get the spike times.
            t = self.spike_times[spike_ids]
            # Keep the spikes belonging to the chunks.
            if subset_chunks:
                spike_ids = spike_ids[_times_in_chunks(t, self.chunks_kept)]
            # Keep spikes from a given subset.
            if subset_spikes is not None:
                spike_ids = np.intersect1d(spike_ids, subset_spikes)
            # Make a subselection if needed.
            if n_spk_clu is not None and n_spk_clu > 0 and len(spike_ids) > n_spk_clu:
                spike_ids = np.random.choice(spike_ids, n_spk_clu, replace=False)
            selection[cluster] = spike_ids
        # Return the concatenation of all spikes.
        return _flatten_per_cluster(selection)
