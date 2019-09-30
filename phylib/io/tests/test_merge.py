# -*- coding: utf-8 -*-

"""Test probe merging."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..merge import Merger
from phylib.io.alf import EphysAlfCreator
from phylib.io.model import load_model
from phylib.io.tests.conftest import _make_dataset
from phylib.io.array import _index_of


#------------------------------------------------------------------------------
# Merging tests
#------------------------------------------------------------------------------

def test_probe_merge_1(tempdir):
    out_dir = tempdir / 'merged'

    # Create two identical datasets.
    probe_names = ('probe_left', 'probe_right')
    for name in probe_names:
        (tempdir / name).mkdir(exist_ok=True, parents=True)
        _make_dataset(tempdir / name, param='dense', has_spike_attributes=False)

    subdirs = [tempdir / name for name in probe_names]

    # Merge them.
    m = Merger(subdirs, out_dir)
    single = load_model(tempdir / probe_names[0] / 'params.py')

    # Test the merged dataset.
    merged = m.merge()
    for name in ('n_spikes', 'n_channels', 'n_templates'):
        assert getattr(merged, name) == getattr(single, name) * 2
    assert merged.sample_rate == single.sample_rate


def test_probe_merge_2(tempdir):
    out_dir = tempdir / 'merged'

    # Create two identical datasets.
    probe_names = ('probe_left', 'probe_right')
    for name in probe_names:
        (tempdir / name).mkdir(exist_ok=True, parents=True)
        _make_dataset(tempdir / name, param='dense', has_spike_attributes=False)
    subdirs = [tempdir / name for name in probe_names]

    # Add small shift in the spike times of the second probe.
    single = load_model(tempdir / probe_names[0] / 'params.py')
    st_path = tempdir / 'probe_right/spike_times.npy'
    np.save(st_path, single.spike_samples + 1)
    # make amplitudes unique and growing so they can serve as key and sorting indices
    single.amplitudes = np.linspace(5, 15, single.n_spikes)
    for m, subdir in enumerate(subdirs):
        np.save(subdir / 'amplitudes.npy', single.amplitudes + 20 * m)

    # Merge them.
    m = Merger(subdirs, out_dir)
    merged = m.merge()

    # Test the merged dataset.
    for name in ('n_spikes', 'n_channels', 'n_templates'):
        assert getattr(merged, name) == getattr(single, name) * 2
    assert merged.sample_rate == single.sample_rate

    # Check the spikes.
    single = load_model(tempdir / probe_names[0] / 'params.py')

    def test_merged_single(merged):
        _, im1, i1 = np.intersect1d(merged.amplitudes, single.amplitudes, return_indices=True)
        _, im2, i2 = np.intersect1d(merged.amplitudes, single.amplitudes + 20, return_indices=True)
        # intersection spans the full vector
        assert i1.size + i2.size == merged.amplitudes.size
        # test spikes
        assert np.allclose(merged.spike_times[im1], single.spike_times[i1])
        assert np.allclose(merged.spike_times[im2], single.spike_times[i2] + 4e-5)
        # the empty clusters are discarded during the merge or alf export
        c1 = _index_of(single.spike_clusters[i1], np.unique(single.spike_clusters[i1]))
        c2 = _index_of(single.spike_clusters[i2], np.unique(single.spike_clusters[i2]))
        assert np.allclose(merged.spike_clusters[im2], c2 + 62)
        assert np.allclose(merged.spike_clusters[im1], c1)
        # test clusters indices indexing via probes
        spike_probes = merged.cluster_probes[merged.spike_clusters]
        assert np.all(np.where(spike_probes == 0) == im1)
        assert np.all(np.where(spike_probes == 1) == im2)
        assert np.all(merged.amplitudes[spike_probes == 0] <= 15)
        assert np.all(merged.amplitudes[spike_probes == 1] >= 20)

    # Convert into ALF and load.
    alf = EphysAlfCreator(merged).convert(tempdir / 'alf')
    test_merged_single(merged)
    test_merged_single(alf)
