# -*- coding: utf-8 -*-

"""py.test utilities."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
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


class TemporaryDirectory_(TemporaryDirectory):
    """HACK: fix on Windows with permission errors when deleting temporary directories."""
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super(TemporaryDirectory_, self).__exit__(exc_type, exc_val, exc_tb)
        except PermissionError as e:  # pragma: no cover
            logger.warning("Permission error while deleting the temporary directory: %s", str(e))


@fixture
def tempdir():
    curdir = os.getcwd()
    with TemporaryDirectory_() as tempdir:
        os.chdir(tempdir)
        yield Path(tempdir)
        os.chdir(curdir)
