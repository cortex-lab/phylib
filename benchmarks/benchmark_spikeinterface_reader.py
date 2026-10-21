"""Opt-in benchmark for ``SpikeInterfaceEphysReader`` prefetching.

Run from the repository root, with spikeinterface installed, with::

    python benchmarks/benchmark_spikeinterface_reader.py

A stand-in recording sleeps for a fixed latency plus a cost per second of data, as a remote
preprocessed recording would. Two access patterns are timed with and without prefetching:
scrolling, which steps a trace window by 10% and pauses as a GUI would to draw it, and
waveform snippets at random times. The benchmark has no timing assertions.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter, sleep

import numpy as np

# Make the documented direct invocation use the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phylib.io.traces import SpikeInterfaceEphysReader  # noqa: E402


class _SlowRecording:
    def __init__(self, n_channels, sample_rate, latency, cost_per_second):
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.latency = latency
        self.cost_per_second = cost_per_second
        self.calls = 0
        self.seconds = 0.

    def get_traces(self, start_frame=None, end_frame=None, segment_index=0):
        duration = (end_frame - start_frame) / self.sample_rate
        self.calls += 1
        self.seconds += duration
        sleep(self.latency + self.cost_per_second * duration)
        return np.zeros((end_frame - start_frame, self.n_channels), dtype=np.float32)


def _reader(args, prefetch):
    import spikeinterface as si
    recording = si.generate_recording(
        num_channels=args.n_channels, sampling_frequency=args.sample_rate,
        durations=[args.duration])
    traces = SpikeInterfaceEphysReader(recording, prefetch=prefetch)
    traces.recording = _SlowRecording(
        args.n_channels, args.sample_rate, args.latency, args.cost_per_second)
    return traces


def _scroll(traces, args):
    window = int(.25 * args.sample_rate)
    start = int(args.duration * args.sample_rate) // 3
    latencies = []
    for _ in range(args.steps):
        t0 = perf_counter()
        traces[start:start + window]
        latencies.append(perf_counter() - t0)
        sleep(args.pause)
        start += window // 10
    return latencies


def _snippets(traces, args):
    rng = np.random.default_rng(0)
    n_samples = int(args.duration * args.sample_rate)
    starts = np.sort(rng.integers(0, n_samples - 82, args.n_snippets))
    latencies = []
    for start in starts:
        t0 = perf_counter()
        traces[start:start + 82]
        latencies.append(perf_counter() - t0)
    return latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-channels', type=int, default=384)
    parser.add_argument('--sample-rate', type=float, default=30000.)
    parser.add_argument('--duration', type=float, default=3600., help="seconds of data")
    parser.add_argument('--latency', type=float, default=.2, help="seconds per get_traces call")
    parser.add_argument(
        '--cost-per-second', type=float, default=.5, help="seconds per second of data read")
    parser.add_argument('--steps', type=int, default=100, help="scroll steps")
    parser.add_argument('--pause', type=float, default=.1, help="seconds between scroll steps")
    parser.add_argument('--n-snippets', type=int, default=100)
    args = parser.parse_args()

    print("pattern    prefetch  total     mean read  reads > 0.1 s  get_traces  data computed")
    for name, run in (('scroll', _scroll), ('snippets', _snippets)):
        for prefetch in (False, True):
            traces = _reader(args, prefetch)
            t0 = perf_counter()
            latencies = run(traces, args)
            total = perf_counter() - t0
            slow = traces.recording
            traces.close()
            print(
                f"{name:<10} {str(prefetch):<8}  {total:>7.2f}s  {np.mean(latencies):>8.3f}s  "
                f"{np.sum(np.array(latencies) > .1):>13}  {slow.calls:>10}  "
                f"{slow.seconds:>11.1f}s")


if __name__ == '__main__':
    main()
