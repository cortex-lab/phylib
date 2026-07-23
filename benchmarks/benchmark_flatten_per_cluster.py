"""Opt-in benchmark for ``_flatten_per_cluster``.

Run from the repository root with::

    python benchmarks/benchmark_flatten_per_cluster.py

The benchmark alternates the legacy and current implementations on every
repeat to reduce ordering bias. It checks exact results but deliberately has
no timing assertions.
"""

import argparse
import gc
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

# Make the documented direct invocation use the repository checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phylib.io.array import _flatten_per_cluster  # noqa: E402


def _legacy_flatten(per_cluster):
    return np.unique(np.concatenate(list(per_cluster.values()))).astype(np.int64)


def _time_once(function, value):
    start = perf_counter()
    result = function(value)
    return perf_counter() - start, result


def _benchmark_case(per_cluster, repeat):
    timings = {'legacy': [], 'current': []}
    functions = {'legacy': _legacy_flatten, 'current': _flatten_per_cluster}
    expected = _legacy_flatten(per_cluster)

    # Warm caches and one-time NumPy dispatch paths.
    np.testing.assert_array_equal(_flatten_per_cluster(per_cluster), expected)
    gc.collect()

    for iteration in range(repeat):
        order = ('legacy', 'current') if iteration % 2 == 0 else ('current', 'legacy')
        for name in order:
            elapsed, result = _time_once(functions[name], per_cluster)
            np.testing.assert_array_equal(result, expected)
            timings[name].append(elapsed)
    return {name: float(np.median(values)) for name, values in timings.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=(10_000, 100_000, 1_000_000))
    parser.add_argument('--repeat', type=int, default=7)
    args = parser.parse_args()

    print("case       size        legacy       current       old/new")
    for size in args.sizes:
        sorted_unique = {0: np.arange(size, dtype=np.int64)}
        # Multiple arrays are intentionally ineligible for the fast path.
        fallback = {
            0: np.arange(0, size, 2, dtype=np.int64),
            1: np.arange(1, size, 2, dtype=np.int64),
        }
        for name, value in (('fast', sorted_unique), ('fallback', fallback)):
            timings = _benchmark_case(value, args.repeat)
            ratio = timings['legacy'] / timings['current']
            print(
                f"{name:<10} {size:>8,}  {timings['legacy']:>10.6f}s  "
                f"{timings['current']:>10.6f}s  {ratio:>9.2f}x")


if __name__ == '__main__':
    main()
