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

# import numpy as np
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
