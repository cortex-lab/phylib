# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
import warnings

import matplotlib
import numpy as np
from pytest import fixture

from phylib import add_default_handler


#------------------------------------------------------------------------------
# Common fixtures
#------------------------------------------------------------------------------

logger = logging.getLogger('phylib')
logger.setLevel(10)
add_default_handler(5, logger=logger)

# Fix the random seed in the tests.
np.random.seed(2015)

warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@fixture
def tempdir(tmp_path):
    # Use the pytest fixture, which allows for --basetemp option to specify a specific temporary
    # directory when running the test suite.
    curdir = os.getcwd()
    os.chdir(tmp_path)
    yield Path(tmp_path)
    os.chdir(curdir)
