# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
import tempfile
import shutil
import warnings

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

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


@fixture
def tempdir():
    curdir = os.getcwd()
    tempdir = tempfile.mkdtemp()
    os.chdir(tempdir)
    yield Path(tempdir)
    os.chdir(curdir)
    shutil.rmtree(tempdir)
