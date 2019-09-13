# -*- coding: utf-8 -*-

"""Test probe merging."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np

from ..merge import Merger
from phylib.io.alf import EphysAlfCreator
from phylib.io.array import _index_of
from phylib.io.model import load_model
from phylib.io.tests.conftest import _make_dataset


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

    # Add small shift in the spike times of the second probe.
    single = load_model(tempdir / probe_names[0] / 'params.py')
    st_path = tempdir / 'probe_right/spike_times.npy'
    np.save(st_path, single.spike_samples + 1)

    subdirs = [tempdir / name for name in probe_names]

    # Merge them.
    m = Merger(subdirs, out_dir)
    merged = m.merge()

    # Test the merged dataset.
    for name in ('n_spikes', 'n_channels', 'n_templates'):
        assert getattr(merged, name) == getattr(single, name) * 2
    assert merged.sample_rate == single.sample_rate

    # Check the spikes.
    single = load_model(tempdir / probe_names[0] / 'params.py')
    assert np.allclose(single.spike_times, merged.spike_times[::2])
    assert np.allclose(merged.spike_times[1::2], single.spike_times + 4e-5)

    # Convert into ALF and load.
    alf = EphysAlfCreator(merged).convert(tempdir / 'alf')

    # Check that (almost) spike cluster probes are interleaved.
    cluster_ids = np.unique(alf.spike_clusters)
    sc_rel = _index_of(alf.spike_clusters, cluster_ids)
    sp = alf.cluster_probes[sc_rel]
    assert (sp[::2] != 0).sum() <= 1
    assert (sp[1::2] != 1).sum() <= 1
