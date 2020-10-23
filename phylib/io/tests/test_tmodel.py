# -*- coding: utf-8 -*-

"""Test template model."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
# import numpy.random as npr
# from numpy.testing import assert_allclose as ac
# from pytest import raises

from .conftest import Dataset
# from phylib.utils import Bunch
from ..loader import TemplateLoaderKS2  # , TemplateLoaderAlf
from ..tmodel import TModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test template model
#------------------------------------------------------------------------------

class TemplateModelDenseTests(unittest.TestCase):
    param = 'dense'
    _loader_cls = TemplateLoaderKS2

    @ classmethod
    def setUpClass(cls):
        cls.ibl = cls.param in ('ks2', 'alf')
        cls.tempdir = Path(tempfile.mkdtemp())
        cls.dset = Dataset(cls.tempdir, cls.param)
        cls.loader = cls._loader_cls()
        cls.loader.open(cls.tempdir)
        cls.model = TModel(cls.loader)

    @ classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tempdir)

    def test_1(self):
        assert len(self.model.template_ids) >= 50
        assert len(self.model.template_ids) == self.model.n_templates

        for tid in self.model.template_ids:
            template = self.model.template(tid)
            assert template.n_spikes > 0
            assert template.amplitude > 0
            assert len(template.channel_ids) > 0

            # Template waveforms.
            w = template.waveforms
            amps = w.max(axis=0) - w.min(axis=0)
            assert np.all(np.diff(amps) <= 0)

    def test_2(self):
        assert len(self.model.cluster_ids) >= 50
        assert len(self.model.cluster_ids) == self.model.n_clusters

        for cid in self.model.cluster_ids:
            cluster = self.model.cluster(cid)
            assert cluster.n_spikes > 0
            assert cluster.amplitude > 0
            assert len(cluster.channel_ids) > 0

            # Cluster waveforms.
            w = cluster.waveforms
            amps = w.max(axis=0) - w.min(axis=0)
            assert np.all(np.diff(amps) <= 0)

            if cid in self.model.template_ids:
                assert cluster.template_ids == [cid]
                assert cluster.template_counts == [len(self.model.spc[cid])]

    def test_3(self):
        for cid in self.model.cluster_ids:
            n = self.model.cluster(cid).n_spikes
            assert self.model.spike_amps(cid).shape == (n,)
            assert self.model.spike_depths(cid).shape == (n,)

            # TODO: proper tests
            self.model.mean_spike_waveforms(cid)
            self.model.spike_waveforms(cid)
            self.model.pc_features(cid)
            self.model.template_features(cid)

    def test_4(self):
        assert self.model.channel_map.shape == (self.model.n_channels,)
        assert self.model.channel_positions.shape == (self.model.n_channels, 2)
        if self.model.channel_shanks is not None:
            assert self.model.channel_shanks.shape == (self.model.n_channels,)
        if self.model.channel_probes is not None:
            assert self.model.channel_probes.shape == (self.model.n_channels,)
