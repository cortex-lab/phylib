# -*- coding: utf-8 -*-

"""Test geometry utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from numpy.testing import assert_allclose as ac
import pytest

from ..geometry import (
    linear_positions,
    staggered_positions,
    _get_data_bounds,
    _boxes_overlap,
    _binary_search,
    _find_box_size,
    get_non_overlapping_boxes,
    get_closest_box,
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


def test_find_box_size():
    x = np.zeros(5)
    y = np.linspace(-1, 1, 5)
    w, h = _find_box_size(x, y, margin=0)
    ac(w, .5, atol=1e-8)
    ac(h, .25, atol=1e-8)


@pytest.mark.parametrize('n_channels', [5, 500])
def test_get_non_overlapping_boxes_1(n_channels):
    x = np.zeros(n_channels)
    y = np.linspace(-1, 1, n_channels)
    box_pos, box_size = get_non_overlapping_boxes(np.c_[x, y])
    ac(box_pos[:, 0], 0, atol=1e-8)
    ac(box_pos[:, 1], -box_pos[::-1, 1], atol=1e-8)

    assert box_size[0] >= .8

    s = np.array([box_size])
    box_bounds = np.c_[box_pos - s, box_pos + s]
    assert box_bounds.min() >= -1
    assert box_bounds.max() <= +1


def test_get_non_overlapping_boxes_2():
    pos = staggered_positions(32)
    box_pos, box_size = get_non_overlapping_boxes(pos)
    assert box_size[0] >= .05

    s = np.array([box_size])
    box_bounds = np.c_[box_pos - s, box_pos + s]
    assert box_bounds.min() >= -1
    assert box_bounds.max() <= +1


def test_get_closest_box():
    n = 10
    px = np.zeros(n)
    py = np.linspace(-1, 1, n)
    box_pos = np.c_[px, py]
    w, h = (1, .9 / n)
    expected = []
    for x in (0, -1, 1, -2, +2):
        for i in range(n):
            expected.extend([
                (x, py[i], i),
                (x, py[i] - h, i),
                (x, py[i] + h, i),
                (x, py[i] - 1.25 * h, max(0, min(i - 1, n - 1))),
                (x, py[i] + 1.25 * h, max(0, min(i + 1, n - 1))),
            ])
    for x, y, i in expected:
        assert get_closest_box((x, y), box_pos, (w, h)) == i


def test_positions():
    probe = staggered_positions(31)
    assert probe.shape == (31, 2)
    ae(probe[-1], (0, 0))

    probe = linear_positions(29)
    assert probe.shape == (29, 2)
