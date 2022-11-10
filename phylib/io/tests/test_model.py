# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import raises

# from phylib.utils import Bunch
from phylib.utils.testing import captured_output
from ..model import from_sparse, load_model

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_from_sparse():
    data = np.array([[0, 1, 2], [3, 4, 5]])
    cols = np.array([[20, 23, 21], [21, 19, 22]])

    def _test(channel_ids, expected):
        expected = np.asarray(expected)
        dense = from_sparse(data, cols, np.array(channel_ids))
        assert dense.shape == expected.shape
        ae(dense, expected)

    _test([0], np.zeros((2, 1)))
    _test([19], [[0], [4]])
    _test([20], [[0], [0]])
    _test([21], [[2], [3]])

    _test([19, 21], [[0, 2], [4, 3]])
    _test([21, 19], [[2, 0], [3, 4]])

    with raises(NotImplementedError):
        _test([19, 19], [[0, 0], [4, 4]])


def test_model_1(template_model_full):
    with captured_output() as (stdout, stderr):
        template_model_full.describe()
    out = stdout.getvalue()
    assert 'sim_binary.dat' in out
    assert '64' in out


def test_model_2(template_model_full):
    m = template_model_full
    tmp = m.get_template(3)
    channel_ids = tmp.channel_ids
    spike_ids = m.get_cluster_spikes(3)

    w = m.get_waveforms(spike_ids, channel_ids)
    assert w is None or w.shape == (len(spike_ids), tmp.template.shape[0], len(channel_ids))

    f = m.get_features(spike_ids, channel_ids)
    assert f is None or f.shape == (len(spike_ids), len(channel_ids), 3)

    tf = m.get_template_features(spike_ids)
    assert tf is None or tf.shape == (len(spike_ids), m.n_templates)


def test_model_3(template_model_full):
    m = template_model_full

    spike_ids = m.get_template_spikes(3)
    n_spikes = len(spike_ids)

    channel_ids = m.get_template_channels(3)
    n_channels = len(channel_ids)

    waveforms = m.get_template_spike_waveforms(3)
    if waveforms is not None:
        assert waveforms.ndim == 3
        assert waveforms.shape[0] == n_spikes
        assert waveforms.shape[2] == n_channels

    tw = m.get_template_waveforms(3)
    assert tw.ndim == 2
    assert tw.shape[1] == n_channels


def test_model_4(template_model_full):
    m = template_model_full

    spike_ids = m.get_cluster_spikes(3)
    n_spikes = len(spike_ids)

    channel_ids = m.get_cluster_channels(3)
    n_channels = len(channel_ids)

    waveforms = m.get_cluster_spike_waveforms(3)
    if waveforms is not None:
        assert waveforms.ndim == 3
        assert waveforms.shape[0] == n_spikes
        assert waveforms.shape[2] == n_channels

    mean_waveforms = m.get_cluster_mean_waveforms(3)
    assert mean_waveforms.mean_waveforms.shape[1] == len(mean_waveforms.channel_ids)


def test_model_depth(template_model):
    depths = template_model.get_depths()
    assert depths.shape == (template_model.n_spikes,)


def test_model_merge(template_model_full):
    m = template_model_full

    # This is the case where we can do the merging
    if not np.all(m.spike_templates == m.spike_clusters) and m.sparse_clusters.cols is None:
        assert len(m.merge_map) > 0
        assert not np.array_equal(m.sparse_clusters.data, m.sparse_templates.data)
        assert m.sparse_clusters.data.shape[0] == m.n_clusters
        assert m.sparse_templates.data.shape[0] == m.n_templates

    else:
        assert len(m.merge_map) == 0
        assert np.array_equal(m.sparse_clusters.data, m.sparse_templates.data)
        assert np.array_equal(m.n_templates, m.n_clusters)


def test_model_save(template_model_full):
    m = template_model_full
    m.save_metadata('test', {1: 1})
    m.save_spike_clusters(m.spike_clusters)


def test_model_spike_waveforms(template_path_full):
    model = load_model(template_path_full)

    if model.traces is not None:
        traces = model.traces[:]
        assert isinstance(traces, np.ndarray)

    waveforms = {}
    for tid in model.template_ids:
        spike_ids = model.get_template_spikes(tid)
        channel_ids = model.get_template_channels(tid)
        waveforms[tid] = model.get_waveforms(spike_ids, channel_ids)

    # Export the waveforms.
    model.save_spikes_subset_waveforms(1000, 16)
    # Fill spike_waveforms after saving them.
    model.spike_waveforms = model._load_spike_waveforms()

    # Check the waveforms loaded from the spike subset waveforms arrays.
    nsw = model.n_samples_waveforms // 2
    if model.spike_waveforms is None:
        return
    for tid in model.template_ids:
        spike_ids = model.get_template_spikes(tid)
        channel_ids = model.get_template_channels(tid)
        spike_ids = np.intersect1d(spike_ids, model.spike_waveforms.spike_ids)
        w = model.get_waveforms(spike_ids, channel_ids)

        # Check the 2 ways of getting the waveforms.
        ae(w, waveforms[tid])

        if model.traces is not None:
            # Check each array with the ground truth, obtained from the raw data.
            for i, spike in enumerate(spike_ids):
                t = int(model.spike_samples[spike])
                wt = traces[t - nsw:t + nsw, channel_ids]

                ae(waveforms[tid][i], wt)
                ae(w[i], wt)

    model.close()


def test_model_metadata_1(template_model_full):
    m = template_model_full

    assert m.metadata.get('group', {}).get(4, None) == 'good'
    assert m.metadata.get('unknown', {}).get(4, None) is None

    assert m.metadata.get('quality', {}).get(6, None) is None
    m.save_metadata('quality', {6: 3})
    m.metadata = m._load_metadata()
    assert m.metadata.get('quality', {}).get(6, None) == 3


def test_model_metadata_2(template_model):
    m = template_model

    m.save_metadata('quality', {0: None, 1: 1})
    m.metadata = m._load_metadata()
    assert m.metadata.get('quality', {}).get(0, None) is None
    assert m.metadata.get('quality', {}).get(1, None) == 1


def test_model_metadata_3(template_model):
    m = template_model

    assert m.metadata.get('met1', {}).get(2, None) == 123.4
    assert m.metadata.get('met1', {}).get(3, None) == 5.678
    assert m.metadata.get('met1', {}).get(4, None) is None
    assert m.metadata.get('met1', {}).get(5, None) is None

    assert m.metadata.get('met2', {}).get(2, None) == 'hello world 1'
    assert m.metadata.get('met2', {}).get(3, None) is None
    assert m.metadata.get('met2', {}).get(4, None) is None
    assert m.metadata.get('met2', {}).get(5, None) == 'hello world 2'


def test_model_spike_attributes(template_model_full):
    model = template_model_full
    assert set(model.spike_attributes.keys()) == set(('randn', 'works'))
    assert model.spike_attributes.works.shape == (model.n_spikes,)
    assert model.spike_attributes.randn.shape == (model.n_spikes, 2)
