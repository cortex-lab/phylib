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
    # single.spike_clusters[single.spike_clusters == 0] = 12
    for m, subdir in enumerate(subdirs):
        np.save(subdir / 'amplitudes.npy', single.amplitudes + 20 * m)
        np.save(subdir / 'spike_clusters.npy', single.spike_clusters)

    # Merge them.
    m = Merger(subdirs, out_dir)
    merged = m.merge()

    # Test the merged dataset.
    for name in ('n_spikes', 'n_channels', 'n_templates'):
        assert getattr(merged, name) == getattr(single, name) * 2
    assert merged.sample_rate == single.sample_rate

    # Check the spikes.
    single = load_model(tempdir / probe_names[0] / 'params.py')

    def test_merged_single(merged, merged_original_amps=None):
        if merged_original_amps is None:
            merged_original_amps = merged.amplitudes
        _, im1, i1 = np.intersect1d(merged_original_amps, single.amplitudes, return_indices=True)
        _, im2, i2 = np.intersect1d(merged_original_amps, single.amplitudes + 20,
                                    return_indices=True)
        # intersection spans the full vector
        assert i1.size + i2.size == merged.amplitudes.size
        # test spikes
        assert np.allclose(merged.spike_times[im1], single.spike_times[i1])
        assert np.allclose(merged.spike_times[im2], single.spike_times[i2] + 4e-5)
        # test clusters
        assert np.allclose(merged.spike_clusters[im2], single.spike_clusters[i2] + 64)
        assert np.allclose(merged.spike_clusters[im1], single.spike_clusters[i1])
        # test templates
        assert np.all(merged.spike_templates[im1] - single.spike_templates[i1] == 0)
        assert np.all(merged.spike_templates[im2] - single.spike_templates[i2] == 64)
        # test probes
        assert np.all(merged.channel_probes == np.r_[single.channel_probes,
                                                     single.channel_probes + 1])
        assert np.all(merged.templates_channels[merged.templates_probes == 0] < single.n_channels)
        assert np.all(merged.templates_channels[merged.templates_probes == 1] >= single.n_channels)
        spike_probes = merged.templates_probes[merged.spike_templates]

        assert np.all(merged_original_amps[spike_probes == 0] <= 15)
        assert np.all(merged_original_amps[spike_probes == 1] >= 20)

        # np.all(merged.sparse_templates.data[:64, :, 0:32] == single.sparse_templates.data)

    # Convert into ALF and load.
    alf = EphysAlfCreator(merged).convert(tempdir / 'alf')
    test_merged_single(merged)
    test_merged_single(alf, merged_original_amps=merged.amplitudes)

    # specific test channel ids only for ALF merge dataset: the raw indices are still individual
    # file indices, the merged channel mapping is in `channels._phy_ids.npy`
    chid = np.load(tempdir.joinpath('alf', 'channels.rawInd.npy'))
    assert np.all(chid == np.r_[single.channel_mapping, single.channel_mapping])

    out_files = list(tempdir.joinpath('alf').glob('*.*'))
    cl_shape = [np.load(f).shape[0] for f in out_files if f.name.startswith('clusters.') and
                f.name.endswith('.npy')]
    sp_shape = [np.load(f).shape[0] for f in out_files if f.name.startswith('spikes.')]
    ch_shape = [np.load(f).shape[0] for f in out_files if f.name.startswith('channels.')]
    assert len(set(cl_shape)) == 1
    assert len(set(sp_shape)) == 1
    assert len(set(ch_shape)) == 1
