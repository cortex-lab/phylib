# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import numpy as np
from numpy.random import uniform
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


#------------------------------------------------------------------------------
# Random colors
#------------------------------------------------------------------------------

def _random_color(h_range=(0., 1.),
                  s_range=(.5, 1.),
                  v_range=(.5, 1.),
                  ):
    """Generate a random RGB color."""
    h, s, v = uniform(*h_range), uniform(*s_range), uniform(*v_range)
    r, g, b = hsv_to_rgb(np.array([[[h, s, v]]])).flat
    return r, g, b


def _is_bright(rgb):
    """Return whether a RGB color is bright or not.
    see https://stackoverflow.com/a/3943023/1595060
    """
    L = 0
    for c, coeff in zip(rgb, (0.2126, 0.7152, 0.0722)):
        if c <= 0.03928:
            c = c / 12.92
        else:
            c = ((c + 0.055) / 1.055) ** 2.4
        L += c * coeff
    if (L + 0.05) / (0.0 + 0.05) > (1.0 + 0.05) / (L + 0.05):
        return True


def _random_bright_color():
    """Generate a random bright color."""
    rgb = _random_color()
    while not _is_bright(rgb):
        rgb = _random_color()
    return rgb


def _hex_to_triplet(h):
    # Convert an hexadecimal color to a triplet of int8 integers.
    if h.startswith('#'):
        h = h[1:]
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


#------------------------------------------------------------------------------
# Colormap
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
# see https://colorcet.pyviz.org/user_guide/Categorical.html
_COLORMAP = np.array([_hex_to_triplet(h) for h in cc.glasbey_light])
_COLORMAP[[0, 1, 2, 3, 4]] = _COLORMAP[[3, 0, 4, 1, 2]]


def _apply_color_masks(color, masks=None, alpha=None):
    alpha = alpha or .5
    hsv = rgb_to_hsv(color[:, :3])
    # Change the saturation and value as a function of the mask.
    if masks is not None:
        hsv[:, 1] *= masks
        hsv[:, 2] *= .5 * (1. + masks)
    color = hsv_to_rgb(hsv)
    n = color.shape[0]
    color = np.c_[color, alpha * np.ones((n, 1))]
    return color


def _colormap(i, alpha=None):
    n = len(_COLORMAP)
    color = tuple(_COLORMAP[i % n] / 255.)
    if alpha is None:
        return color
    else:
        assert 0 <= alpha <= 1
        return color + (alpha,)


def _spike_colors(spike_clusters=None, masks=None, alpha=None):
    n = len(_COLORMAP)
    if spike_clusters is not None:
        c = _COLORMAP[np.mod(spike_clusters, n), :] / 255.
    else:
        c = np.ones((masks.shape[0], 3))
    c = _apply_color_masks(c, masks=masks, alpha=alpha)
    return c


class ColorSelector(object):
    """Return the color of a cluster.

    If the cluster belongs to the selection, returns the colormap color.

    Otherwise, return a random color and remember this color.

    """
    def __init__(self):
        self._colors = {}

    def get(self, clu, cluster_ids=None, cluster_group=None, alpha=None):
        alpha = alpha or .5
        if cluster_group in ('noise', 'mua'):
            color = (.5,) * 4
        elif cluster_ids and clu in cluster_ids:
            i = cluster_ids.index(clu)
            color = _colormap(i, alpha)
        else:
            if clu in self._colors:
                return self._colors[clu]
            color = _random_color(v_range=(.5, .75))
            color = tuple(color) + (alpha,)
            self._colors[clu] = color
        assert len(color) == 4
        return color
