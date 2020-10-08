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
import numpy.random as npr
from numpy.testing import assert_allclose as ac
from pytest import raises

from .conftest import Dataset
from phylib.utils import Bunch
from ..tmodel import TModel

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Test template model
#------------------------------------------------------------------------------

def test_tmodel_1():
    pass
