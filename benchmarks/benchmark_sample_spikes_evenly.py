"""Opt-in benchmark for ``_sample_spikes_evenly``.

Run from the repository root with::

    python benchmarks/benchmark_sample_spikes_evenly.py

The sample budget stays fixed while the source size grows, demonstrating that
the sampler does not materialize work proportional to the cluster size.
"""

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phylib.io.array import _sample_spikes_evenly  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=(10_000, 100_000, 1_000_000))
    parser.add_argument('--sample-size', type=int, default=10_000)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()

    print('source size  sample size  median time')
    for size in args.sizes:
        spike_ids = np.arange(size, dtype=np.int64)
        sample_size = min(args.sample_size, size)
        _sample_spikes_evenly(spike_ids, sample_size)  # Warm NumPy dispatch.
        timings = []
        for _ in range(args.repeat):
            start = perf_counter()
            sampled = _sample_spikes_evenly(spike_ids, sample_size)
            timings.append(perf_counter() - start)
        assert len(sampled) == sample_size
        print(f'{size:>11,}  {sample_size:>11,}  {np.median(timings):>10.6f}s')


if __name__ == '__main__':
    main()
