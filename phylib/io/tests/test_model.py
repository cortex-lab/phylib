# -*- coding: utf-8 -*-

"""Testing the Template m."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import unittest
from pathlib import Path
import tempfile
import shutil

import numpy as np
from numpy.testing import assert_equal as ae
from pytest import raises

# from phylib.utils import Bunch
from phylib.utils.testing import captured_output
from ..model import from_sparse
from .conftest import Dataset


logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Unit tests
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


#------------------------------------------------------------------------------
# Integration tests
#------------------------------------------------------------------------------

def _subset_spikes(m, spike_ids):
    if m.spike_waveforms is not None:
        spike_ids = np.intersect1d(spike_ids, m.spike_waveforms.spike_ids)
    # HACK: take a small subset of the spikes to make the test faster
    spike_ids = spike_ids[m.spike_times[spike_ids] <= 60]
    return spike_ids


class TemplateModelDenseTest(unittest.TestCase):
    param = 'dense'

    @classmethod
    def setUpClass(cls):
        cls.ibl = cls.param in ('ks2', 'alf')
        cls.tempdir = Path(tempfile.mkdtemp())
        cls.dset = Dataset(cls.tempdir, cls.param)
        cls.model = cls.dset.create_model()

    @classmethod
    def tearDownClass(cls):
        cls.dset.destroy_model()
        shutil.rmtree(cls.tempdir)

    def get_spikes_and_channels(self, template_id):
        m = self.model
        tmp = m.get_template(template_id)
        channel_ids = tmp.channel_ids
        spike_ids = m.get_cluster_spikes(template_id)
        spike_ids = _subset_spikes(m, spike_ids)
        return spike_ids, channel_ids

    def test_model_01_describe(self):
        m = self.model
        with captured_output() as (stdout, stderr):
            m.describe()
        out = stdout.getvalue()
        assert '.dat' in out or '.cbin' in out or '.bin' in out
        assert '64' in out

    def test_model_02_template(self):
        m = self.model
        tmp = m.get_template(3)
        spike_ids, channel_ids = self.get_spikes_and_channels(3)
        n_channels = len(channel_ids)

        # Check the template amplitude is decreasing across the reordered channels.
        tmp = m.sparse_templates.data
        amp = tmp.max(axis=1) - tmp.min(axis=1)
        # NOTE: currently ALF datasets do not comply to this convention
        if not self.ibl:
            assert np.all(np.diff(amp, axis=1) <= 0)

        # Template waveforms.
        tw = m.get_template_waveforms(3)
        assert tw.ndim == 2
        assert tw.shape[1] == n_channels

        # Features.
        f = m.get_features(spike_ids, channel_ids)
        assert f is None or f.shape == (len(spike_ids), len(channel_ids), 3)

        # Template features.
        tf = m.get_template_features(spike_ids)
        assert tf is None or tf.shape == (len(spike_ids), m.n_templates)

    def test_model_03_spike_attributes(self):
        m = self.model
        if self.ibl:
            return

        assert set(m.spike_attributes.keys()) == set(('randn', 'works'))
        assert m.spike_attributes.works.shape == (m.n_spikes,)
        assert m.spike_attributes.randn.shape == (m.n_spikes, 2)

    def test_model_04_metadata(self):
        m = self.model

        assert m.metadata.get('met1', {}).get(2, None) == 123.4
        assert m.metadata.get('met1', {}).get(3, None) == 5.678
        assert m.metadata.get('met1', {}).get(4, None) is None
        assert m.metadata.get('met1', {}).get(5, None) is None

        assert m.metadata.get('met2', {}).get(2, None) == 'hello world 1'
        assert m.metadata.get('met2', {}).get(3, None) is None
        assert m.metadata.get('met2', {}).get(4, None) is None
        assert m.metadata.get('met2', {}).get(5, None) == 'hello world 2'

        assert m.metadata.get('quality', {}).get(6, None) is None
        m.save_metadata('quality', {6: 3, 1: None})
        m.metadata = m._load_metadata()
        assert m.metadata.get('quality', {}).get(6, None) == 3
        assert m.metadata.get('quality', {}).get(1, None) is None

    def test_model_04_metadata_misc(self):
        m = self.model
        if not self.ibl:
            assert m.metadata.get('group', {}).get(4, None) == 'good'
            assert m.metadata.get('unknown', {}).get(4, None) is None

    def test_model_05_template_waveforms(self):
        m = self.model
        mean_waveforms = m.get_cluster_mean_waveforms(3)
        assert mean_waveforms.mean_waveforms.shape[1] == len(mean_waveforms.channel_ids)

    def test_model_06_spike_waveforms(self):
        m = self.model
        spike_ids, channel_ids = self.get_spikes_and_channels(3)
        w = m.get_waveforms(spike_ids, channel_ids)
        assert w is None or w.shape == (len(spike_ids), m.n_samples_waveforms, len(channel_ids))
        if w is not None and w.size > 0:
            assert not np.all(w == 0)

    def test_model_07_spike_amplitudes(self):
        if not self.ibl:
            return
        m = self.model
        assert 1e-4 < m.amplitudes.mean() < 1e-2

    def test_model_08_spike_depths(self):
        m = self.model
        depths = m.get_depths()
        if depths is not None:
            assert depths.shape == (m.n_spikes,)

    def test_model_09_features(self):
        m = self.model
        spike_ids, channel_ids = self.get_spikes_and_channels(3)
        features = m.get_features(spike_ids, channel_ids)
        if self.param != 'misc':
            assert features.shape == (len(spike_ids), len(channel_ids), 3)

    def test_model_10_save(self):
        m = self.model
        m.save_metadata('test', {1: 1})
        m.save_spike_clusters(m.spike_clusters)

    # TODO: move to ALF
    # def test_model_11_subset_waveforms(self):
    #     m = self.model
    #     waveforms = {}
    #     tids = m.template_ids
    #     if self.ibl:
    #         tids = tids[::10]
    #     for tid in tids:
    #         spike_ids, channel_ids = self.get_spikes_and_channels(tid)
    #         if len(spike_ids) == 0:
    #             logger.debug("Skip template %d", tid)
    #             continue
    #         logger.debug("Extracting spike waveforms for template %d", tid)
    #         # NOTE: all spikes <60s are selected here as there is not a phy subset waveforms file
    #         # yet
    #         waveforms[tid] = (spike_ids, m.get_waveforms(spike_ids, channel_ids))
    #         # waveforms[tid] = m.get_waveforms(spike_ids, channel_ids)

    #     # Export ALL waveforms.
    #     m.save_spikes_subset_waveforms(0, m.n_channels_loc)
    #     # Fill spike_waveforms after saving them.
    #     m.spike_waveforms = m._load_spike_waveforms()

    #     if m.traces is not None:
    #         traces = m.traces
    #         # assert isinstance(traces, np.ndarray)
    #     if m.traces is None and m.spike_waveforms is None:
    #         return

    #     # Check the waveforms loaded from the spike subset waveforms arrays.
    #     nsw = m.n_samples_waveforms // 2
    #     assert m.spike_waveforms is not None
    #     for tid in tids:
    #         logger.debug("Check template %d", tid)
    #         spike_ids, channel_ids = self.get_spikes_and_channels(tid)
    #         # NOTE: there are less spikes here, as they are <60s, but also they are subset
    #         # from subset waveforms
    #         assert len(channel_ids) <= 32
    #         if len(spike_ids) == 0:
    #             logger.debug("Skip template %d", tid)
    #             continue
    #         w = m.get_waveforms(spike_ids, channel_ids)
    #         assert waveforms[tid].shape[0] == len(spike_ids)

    #         # Check the 2 ways of getting the waveforms.
    #         ae(w, waveforms[tid])

    #         if traces is not None:
    #             # Check each array with the ground truth, obtained from the raw data.
    #             for i, spike in enumerate(spike_ids):
    #                 t = int(m.spike_samples[spike])
    #                 wt = traces[t - nsw:t + nsw, channel_ids]

    #                 ae(waveforms[tid][i], wt)
    #                 ae(w[i], wt)


#------------------------------------------------------------------------------
# Other datasets
#------------------------------------------------------------------------------

class TemplateModelSparseTest(TemplateModelDenseTest):
    param = 'sparse'


class TemplateModelMiscTest(TemplateModelDenseTest):
    param = 'misc'


class TemplateModelKS2Test(TemplateModelDenseTest):
    param = 'ks2'


class TemplateModelALFTest(TemplateModelDenseTest):
    param = 'alf'
