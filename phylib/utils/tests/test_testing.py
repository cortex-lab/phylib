# -*- coding: utf-8 -*-

"""Tests of testing utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from copy import deepcopy
import logging

import numpy as np

from ..testing import captured_output, captured_logging, _assert_equal

logger = logging.getLogger('phylib')


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_logging_1():
    print()
    logger.setLevel(5)
    logger.log(5, "level 5")
    logger.log(10, "debug")
    logger.log(20, "info")
    logger.log(30, "warning")
    logger.log(40, "error")


def test_captured_output():
    with captured_output() as (out, err):
        print('Hello world!')
    assert out.getvalue().strip() == 'Hello world!'


def test_captured_logging():
    handlers = logger.handlers
    with captured_logging() as buf:
        logger.debug('Hello world!')
    assert 'Hello world!' in buf.getvalue()
    assert logger.handlers == handlers


def test_assert_equal():
    d = {'a': {'b': np.random.rand(5), 3: 'c'}, 'b': 2.}
    d_bis = deepcopy(d)
    d_bis['a']['b'] = d_bis['a']['b'] + 1e-16
    _assert_equal(d, d_bis)
