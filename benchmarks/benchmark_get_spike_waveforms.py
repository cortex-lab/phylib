"""Compare the historical and optimized sparse waveform extraction paths.

Run from the repository root, for example:

    python benchmarks/benchmark_get_spike_waveforms.py

This is intentionally a standalone benchmark rather than a timing assertion in
the unit test suite.
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

# Make direct execution from a source checkout independent of installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phylib.io.array import _index_of  # noqa: E402
from phylib.io.traces import (  # noqa: E402
    _get_spike_ids_rel, _get_spike_waveforms_fast, _get_spike_waveforms_legacy,
    get_spike_waveforms)
from phylib.utils import Bunch  # noqa: E402


def get_spike_waveforms_reference(
        spike_ids, channel_ids, spike_waveforms=None, n_samples_waveforms=None):
    """Historical get_spike_waveforms() implementation."""
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


def _median_runtime(func, args, repeat):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        durations.append(time.perf_counter() - start)
    return np.median(durations)


def _resolve_spike_ids_reference(spike_ids, available_spike_ids):
    assert np.all(np.isin(spike_ids, available_spike_ids))
    return _index_of(spike_ids, available_spike_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stored-spikes', type=int, default=5000)
    parser.add_argument('--requested-spikes', type=int, default=1000)
    parser.add_argument('--samples', type=int, default=82)
    parser.add_argument('--sparse-channels', type=int, default=32)
    parser.add_argument('--requested-channels', type=int, default=64)
    parser.add_argument('--total-channels', type=int, default=384)
    parser.add_argument('--spike-id-stride', type=int, default=1000)
    parser.add_argument('--repeat', type=int, default=7)
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    spike_ids = (
        np.arange(args.stored_spikes, dtype=np.int64) * args.spike_id_stride +
        1_000_000)
    # Each row is deliberately unsorted, as real sparse columns are commonly
    # ordered by template amplitude rather than channel ID.
    spike_channels = np.vstack([
        rng.choice(args.total_channels, args.sparse_channels, replace=False)
        for _ in range(args.stored_spikes)
    ]).astype(np.int32)
    waveforms = rng.standard_normal((
        args.stored_spikes, args.samples, args.sparse_channels)).astype(np.float32)
    spike_waveforms = Bunch(
        spike_ids=spike_ids,
        spike_channels=spike_channels,
        waveforms=waveforms,
    )
    requested_spikes = rng.choice(
        spike_ids, args.requested_spikes, replace=True)
    requested_channels = rng.choice(
        args.total_channels, args.requested_channels, replace=False)
    call_args = (
        requested_spikes, requested_channels, spike_waveforms, args.samples)

    expected = get_spike_waveforms_reference(*call_args)
    actual = get_spike_waveforms(*call_args)
    np.testing.assert_array_equal(actual, expected)

    # Warm both paths before collecting samples.
    get_spike_waveforms_reference(*call_args)
    get_spike_waveforms(*call_args)
    spike_ids_rel = _get_spike_ids_rel(requested_spikes, spike_ids)
    old_median = _median_runtime(
        get_spike_waveforms_reference, call_args, args.repeat)
    new_median = _median_runtime(
        get_spike_waveforms, call_args, args.repeat)
    old_resolution_median = _median_runtime(
        _resolve_spike_ids_reference, (requested_spikes, spike_ids), args.repeat)
    new_resolution_median = _median_runtime(
        _get_spike_ids_rel, (requested_spikes, spike_ids), args.repeat)
    old_extraction_median = _median_runtime(
        _get_spike_waveforms_legacy,
        (spike_ids_rel, requested_channels, spike_waveforms, args.samples),
        args.repeat)
    new_extraction_median = _median_runtime(
        _get_spike_waveforms_fast,
        (spike_ids_rel, requested_channels, spike_waveforms, args.samples),
        args.repeat)

    print(
        f"shape: stored={args.stored_spikes}, requested={args.requested_spikes}, "
        f"samples={args.samples}, sparse_channels={args.sparse_channels}, "
        f"requested_channels={args.requested_channels}, "
        f"spike_id_stride={args.spike_id_stride}")
    print(f"old median: {old_median * 1000:.3f} ms")
    print(f"new median: {new_median * 1000:.3f} ms")
    print(f"speedup: {old_median / new_median:.2f}x")
    print(f"old spike-ID validation/index median: {old_resolution_median * 1000:.3f} ms")
    print(f"new spike-ID validation/index median: {new_resolution_median * 1000:.3f} ms")
    print(
        f"spike-ID resolution speedup: "
        f"{old_resolution_median / new_resolution_median:.2f}x")
    print(f"old extraction-only median: {old_extraction_median * 1000:.3f} ms")
    print(f"new extraction-only median: {new_extraction_median * 1000:.3f} ms")
    print(
        f"extraction-only speedup: "
        f"{old_extraction_median / new_extraction_median:.2f}x")


if __name__ == '__main__':
    main()
