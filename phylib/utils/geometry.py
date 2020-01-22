# -*- coding: utf-8 -*-

"""Plotting utilities."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from functools import partial
import logging

import numpy as np

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Common probe layouts
#------------------------------------------------------------------------------

def linear_positions(n_channels):
    """Linear channel positions along the vertical axis."""
    return np.c_[np.zeros(n_channels),
                 np.linspace(0., 1., n_channels)]


def staggered_positions(n_channels):
    """Generate channel positions for a staggered probe."""
    i = np.arange(n_channels - 1)
    x, y = (-1) ** i * (5 + i), 10 * (i + 1)
    pos = np.flipud(np.r_[np.zeros((1, 2)), np.c_[x, y]])
    return pos


#------------------------------------------------------------------------------
# Box positioning
#------------------------------------------------------------------------------

def range_transform(from_bounds, to_bounds, positions, do_offset=True):
    """Transform for a rectangle to another."""
    from_bounds = np.asarray(from_bounds)
    to_bounds = np.asarray(to_bounds)
    positions = np.asarray(positions)

    assert from_bounds.ndim == to_bounds.ndim == positions.ndim == 2

    f0 = from_bounds[..., :2]
    f1 = from_bounds[..., 2:]
    t0 = to_bounds[..., :2]
    t1 = to_bounds[..., 2:]

    # Degenerate axes are extended maximally.
    for z0, z1 in ((f0, f1), (t0, t1)):
        for i in range(2):
            ind = np.abs(z0[:, i] - z1[:, i]) < 1e-8
            z0[ind, i] = -1
            z1[ind, i] = +1

    d = (f1 - f0)
    d[d == 0] = 1

    out = positions.copy()
    if do_offset:
        out -= f0.astype(out.dtype)
    out *= ((t1 - t0) / d).astype(out.dtype)
    if do_offset:
        out += t0.astype(out.dtype)
    return out


def _boxes_overlap(x0, y0, x1, y1):
    """Return whether a set of boxes, defined by their 2D corners, overlap or  not."""
    assert x0.ndim == y0.ndim == y0.ndim == y1.ndim == 2
    n = len(x0)
    overlap_matrix = ((x0 < x1.T) & (x1 > x0.T) & (y0 < y1.T) & (y1 > y0.T))
    overlap_matrix[np.arange(n), np.arange(n)] = False
    return np.any(overlap_matrix.ravel())


def _binary_search(f, xmin, xmax, eps=1e-9):
    """Return the largest x such f(x) is True."""
    middle = (xmax + xmin) / 2.
    while xmax - xmin > eps:
        assert xmin < xmax
        middle = (xmax + xmin) / 2.
        if f(xmax):
            return xmax
        if not f(xmin):
            return xmin
        if f(middle):
            xmin = middle
        else:
            xmax = middle
    return middle


def _find_box_size(x, y, ar=.5, margin=0):
    """Return the maximum (half) box size such that boxes centered around box positions
    do not overlap."""
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    logger.log(5, "Get box size for %d points.", len(x))
    # Deal with degenerate x case.
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        # If all positions are vertical, the width can be maximum.
        wmax = 1.
    else:
        wmax = xmax - xmin

    def f1(w, keep_aspect_ratio=True, h=None):
        """Return true if the configuration with the current box size
        is non-overlapping."""
        # NOTE: w|h are the *half* width|height.
        if keep_aspect_ratio:
            h = w * ar  # fixed aspect ratio
        return not _boxes_overlap(x - w, y - h, x + w, y + h)

    # Find the largest box size leading to non-overlapping boxes.
    w = _binary_search(f1, 0, wmax)
    w = w * (1 - margin)  # margin
    # Clip the half-width.
    h = w * ar  # aspect ratio

    # Extend the boxes horizontally as much as possible.
    w = _binary_search(partial(f1, keep_aspect_ratio=False, h=h), w, wmax)
    w = w * (1 - margin)  # margin

    return w, h


def get_non_overlapping_boxes(box_pos):
    """Normalize box positions and return a convenient half box size."""
    box_pos = np.asarray(box_pos)
    assert box_pos.ndim == 2
    assert box_pos.shape[1] == 2
    # Renormalize box_pos.
    mx, my = box_pos.min(axis=0)
    Mx, My = box_pos.max(axis=0)
    box_pos = range_transform([[mx, my, Mx, My]], [[-1, -1, +1, +1]], box_pos)
    # Compute box size.
    x, y = box_pos.T
    w, h = _find_box_size(x, y, margin=.1)
    # Renormalize again so that the boxes fit inside the view.
    mx, my = np.min(box_pos - np.array([[w, h]]), axis=0)
    Mx, My = np.max(box_pos + np.array([[w, h]]), axis=0)
    b1 = [[mx, my, Mx, My]]
    b2 = [[-1, -1, 1, 1]]
    box_pos = range_transform(b1, b2, box_pos)
    w, h = range_transform(b1, b2, [[w, h]], do_offset=False).ravel()
    w *= .95
    h *= .9
    logger.log(5, "Found box size %s.", (w, h))
    return box_pos, (w, h)


def get_closest_box(pos, box_pos, box_size):
    """Return the box closest to a given point."""
    # box_size is the half size
    # see https://gamedev.stackexchange.com/a/44496
    w, h = box_size
    x, y = pos
    px, py = box_pos.T
    dx = np.maximum(np.abs(px - x) - w, 0)
    dy = np.maximum(np.abs(py - y) - h, 0)
    d = dx * dx + dy * dy
    return np.argmin(d)


#------------------------------------------------------------------------------
# Data bounds utilities
#------------------------------------------------------------------------------

def _get_data_bounds(data_bounds, pos=None, length=None):
    """"Prepare data bounds, possibly using min/max of the data."""
    if data_bounds is None or (isinstance(data_bounds, str) and data_bounds == 'auto'):
        if pos is not None and len(pos):
            m, M = pos.min(axis=0), pos.max(axis=0)
            data_bounds = [m[0], m[1], M[0], M[1]]
        else:
            data_bounds = [-1, -1, 1, 1]
    data_bounds = np.atleast_2d(data_bounds)

    ind_x = data_bounds[:, 0] == data_bounds[:, 2]
    ind_y = data_bounds[:, 1] == data_bounds[:, 3]
    if np.sum(ind_x):
        data_bounds[ind_x, 0] -= 1
        data_bounds[ind_x, 2] += 1
    if np.sum(ind_y):
        data_bounds[ind_y, 1] -= 1
        data_bounds[ind_y, 3] += 1

    # Extend the data_bounds if needed.
    if length is None:
        length = pos.shape[0] if pos is not None else 1
    if data_bounds.shape[0] == 1:
        data_bounds = np.tile(data_bounds, (length, 1))

    # Check the shape of data_bounds.
    assert data_bounds.shape == (length, 4)

    assert data_bounds.ndim == 2
    assert data_bounds.shape[1] == 4
    assert np.all(data_bounds[:, 0] < data_bounds[:, 2])
    assert np.all(data_bounds[:, 1] < data_bounds[:, 3])

    return data_bounds
