# -*- coding: utf-8 -*-

"""Testing the Template model."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import raises

from phylib.utils import Bunch
from phylib.utils.testing import captured_output
from ..model import from_sparse

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


def test_model_1(template_model):
    with captured_output() as (stdout, stderr):
        template_model.describe()
    out = stdout.getvalue()
    print(out)
    assert 'sim_binary.dat' in out
    if '(300000, 32)' not in out:
        assert 'Data shape              None' in out
    assert '64' in out


def test_model_2(template_model):
    m = template_model
    tmp = m.get_template(3)
    channel_ids = tmp.channel_ids
    spike_ids = m.spikes_in_template(3)

    w = m.get_waveforms(spike_ids, channel_ids)
    assert w is None or w.shape == (len(spike_ids), tmp.template.shape[0], len(channel_ids))

    f = m.get_features(spike_ids, channel_ids)
    assert f is None or f.shape == (len(spike_ids), len(channel_ids), 3)

    tf = m.get_template_features(spike_ids)
    assert tf is None or tf.shape == (len(spike_ids), m.n_templates)


def test_model_save(template_model):
    m = template_model
    m.save_metadata('test', {1: 1})
    m.save_spike_clusters(m.spike_clusters)
    m.save_mean_waveforms({1: Bunch(
        channel_ids=np.arange(m.n_channels),
        data=np.zeros((1, m.n_samples_templates, m.n_channels)))})


def test_model_metadata_1(template_model):
    m = template_model
    assert m.metadata_fields

    assert m.get_metadata('group').get(4, None) == 'good'
    assert m.get_metadata('unknown').get(4, None) is None

    assert m.get_metadata('quality').get(6, None) is None
    m.save_metadata('quality', {6: 3})
    m.metadata = m._load_metadata()
    assert m.get_metadata('quality').get(6, None) == '3'


def test_model_metadata_2(template_model):
    m = template_model
    assert m.metadata_fields

    m.save_metadata('quality', {0: None, 1: 1})
    m.metadata = m._load_metadata()
    assert m.get_metadata('quality').get(0, None) is None
    assert m.get_metadata('quality').get(1, None) == '1'
