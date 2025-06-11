# -*- coding: utf-8 -*-

"""Utility functions used for tests."""

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO

from numpy.testing import assert_allclose as ac
from numpy.testing import assert_array_equal as ae

from ._types import _is_array_like

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------


@contextmanager
def captured_output():
    """Context manager that captures all output to stdout and stderr."""
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@contextmanager
def captured_logging(name=None):
    """Context manager that captures all logging."""
    logger = logging.getLogger(name)
    handlers = list(logger.handlers)
    for handler in logger.handlers:
        logger.removeHandler(handler)
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield buffer
    buffer.flush()
    logger.removeHandler(handler)
    for handler in handlers:
        logger.addHandler(handler)
    handler.close()


def _assert_equal(d_0, d_1):
    """Check that two objects are equal."""
    # Compare arrays.
    if _is_array_like(d_0):
        try:
            ae(d_0, d_1)
        except AssertionError:
            ac(d_0, d_1)
    # Compare dicts recursively.
    elif isinstance(d_0, dict):
        assert set(d_0) == set(d_1)
        for k_0 in d_0:
            _assert_equal(d_0[k_0], d_1[k_0])
    else:
        # General comparison.
        assert d_0 == d_1


def _in_travis():  # pragma: no cover
    """Return whether we're in travis."""
    return 'TRAVIS' in os.environ
