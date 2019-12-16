# -*- coding: utf-8 -*-

"""Test geometry utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac

from ..geometry import (
    linear_positions,
    staggered_positions,
    _get_data_bounds,
    _boxes_overlap,
    _binary_search,
)


#------------------------------------------------------------------------------
# Test utilities
#------------------------------------------------------------------------------

def test_get_data_bounds():
    ae(_get_data_bounds(None), [[-1., -1., 1., 1.]])

    db0 = np.array([[0, 1, 4, 5],
                    [0, 1, 4, 5],
                    [0, 1, 4, 5]])
    arr = np.arange(6).reshape((3, 2))
    assert np.all(_get_data_bounds(None, arr) == [[0, 1, 4, 5]])

    db = db0.copy()
    assert np.all(_get_data_bounds(db, arr) == [[0, 1, 4, 5]])

    db = db0.copy()
    db[2, :] = [1, 1, 1, 1]
    assert np.all(_get_data_bounds(db, arr)[:2, :] == [[0, 1, 4, 5]])
    assert np.all(_get_data_bounds(db, arr)[2, :] == [0, 0, 2, 2])

    db = db0.copy()
    db[:2, :] = [1, 1, 1, 1]
    assert np.all(_get_data_bounds(db, arr)[:2, :] == [[0, 0, 2, 2]])
    assert np.all(_get_data_bounds(db, arr)[2, :] == [0, 1, 4, 5])


def test_boxes_overlap():

    def _get_args(boxes):
        x0, y0, x1, y1 = np.array(boxes).T
        x0 = x0[:, np.newaxis]
        x1 = x1[:, np.newaxis]
        y0 = y0[:, np.newaxis]
        y1 = y1[:, np.newaxis]
        return x0, y0, x1, y1

    boxes = [[-1, -1, 0, 0], [0.01, 0.01, 1, 1]]
    x0, y0, x1, y1 = _get_args(boxes)
    assert not _boxes_overlap(x0, y0, x1, y1)

    boxes = [[-1, -1, 0.1, 0.1], [0, 0, 1, 1]]
    x0, y0, x1, y1 = _get_args(boxes)
    assert _boxes_overlap(x0, y0, x1, y1)

    x = np.zeros((5, 1))
    x0 = x - .1
    x1 = x + .1
    y = np.linspace(-1, 1, 5)[:, np.newaxis]
    y0 = y - .2
    y1 = y + .2
    assert not _boxes_overlap(x0, y0, x1, y1)


def test_binary_search():
    def f(x):
        return x < .4
    ac(_binary_search(f, 0, 1), .4)
    ac(_binary_search(f, 0, .3), .3)
    ac(_binary_search(f, .5, 1), .5)


def test_positions():
    probe = staggered_positions(31)
    assert probe.shape == (31, 2)
    ae(probe[-1], (0, 0))

    probe = linear_positions(29)
    assert probe.shape == (29, 2)
