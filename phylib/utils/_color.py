# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import logging

from phylib.io.array import _index_of

import numpy as np
from numpy.random import uniform
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

logger = logging.getLogger(__name__)


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
_COLORMAP[[0, 1, 2, 3, 4, 5]] = _COLORMAP[[3, 0, 4, 5, 2, 1]]
_COLORMAP[0] = [8, 146, 252]
_COLORMAP[1] = [255, 2, 2]


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


def _validate_colormap(colormap):
    assert colormap.ndim == 2
    n = colormap.shape[0]
    if colormap.shape[1] == 3:
        colormap = np.c_[colormap, np.ones((n, 1))]
    assert colormap.shape[1] == 4
    return colormap


def _continuous_colormap(colormap, values, vmin=None, vmax=None):
    colormap = _validate_colormap(colormap)
    n = colormap.shape[0]
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    assert vmin is not None
    assert vmax is not None
    denom = vmax - vmin
    denom = denom if denom != 0 else 1
    i = np.round((n - 1) * (values - vmin) / denom).astype(np.int32)
    return colormap[i, :]


def _categorical_colormap(colormap, values, vmin=None, vmax=None):
    colormap = _validate_colormap(colormap)
    n = colormap.shape[0]
    if vmin is None and vmax is None:
        # Find unique values and keep the order.
        _, idx = np.unique(values, return_index=True)
        lookup = values[np.sort(idx)]
        x = _index_of(values, lookup)
    else:
        x = values
    return colormap[x % n, :]


def _make_colorcet_colormap(name, categorical=False):
    f = _continuous_colormap if not categorical else _categorical_colormap

    def _colormap(values, vmin=None, vmax=None):
        colormap = np.array(getattr(cc, name))
        return f(colormap, values, vmin=vmin, vmax=vmax)

    return _colormap


# Built-in colormaps.
rainbow = _make_colorcet_colormap('rainbow_bgyr_35_85_c73', False)
linear = _make_colorcet_colormap('linear_wyor_100_45_c55', False)
diverging = _make_colorcet_colormap('diverging_linear_bjy_30_90_c45', False)
categorical = _make_colorcet_colormap('glasbey_bw_minc_20_minl_30', True)


def get_colormap(colormap):
    """Get a colormap as a function values => (n, 4) RGBA array."""
    if isinstance(colormap, str):
        return globals().get(colormap, _make_colorcet_colormap(colormap))
    return colormap


class ClusterColorSelector(object):
    """Assign a color to clusters depending on cluster labels or metrics."""
    color_field = 'cluster'
    _colormap = 'categorical'

    def __init__(
            self, cluster_labels=None, cluster_metrics=None,
            field=None, colormap=None, cluster_ids=None):
        self.cluster_ids = None
        self.cluster_labels = cluster_labels or {}
        self.cluster_metrics = cluster_metrics or {}
        # Used to initialize the value range for the clusters.
        assert cluster_ids is not None
        self.cluster_ids = cluster_ids
        self.set_color_field(field=field, colormap=colormap)

    def set_color_field(self, field=None, colormap=None):
        """Set the field used to choose the cluster colors, and the associated colormap."""
        self._colormap = get_colormap(colormap or self._colormap)  # self.colormap is a function
        self.color_field = field or self.color_field
        # Recompute the value range.
        self.set_cluster_ids(self.cluster_ids)

    def set_cluster_ids(self, cluster_ids):
        """Precompute the value range for all clusters."""
        self.cluster_ids = cluster_ids
        values = self.get_values(self.cluster_ids)
        self.vmin, self.vmax = values.min(), values.max()

    def colormap(self, values):
        return self._colormap(values, vmin=self.vmin, vmax=self.vmax)

    def _get_cluster_value(self, cluster_id):
        """Return the field value for a given cluster."""
        field = self.color_field
        if field == 'cluster':
            return cluster_id
        elif field in self.cluster_labels:
            return self.cluster_labels.get(field, {}).get(cluster_id, None)
        elif field in self.cluster_metrics:
            return self.cluster_metrics.get(field, lambda cl: None)(cluster_id)
        logger.warning("The field %s is not an existing cluster label or metrics.", field)
        return 0

    def get(self, cluster_id, alpha=None):
        """Return the color of a given cluster."""
        assert self.cluster_ids is not None
        assert self._colormap is not None
        val = self._get_cluster_value(cluster_id)
        col = self.colormap(np.array([val]))[0].tolist()
        if alpha is not None:
            col[3] = alpha
        return tuple(col)

    def get_values(self, cluster_ids):
        return np.array([self._get_cluster_value(cluster_id) for cluster_id in cluster_ids])

    def get_colors(self, cluster_ids):
        """Return cluster colors using a given field ('cluster', label, or metrics).

        colormap is a function values => colors

        """
        # TODO: categorical values as strings
        values = self.get_values(cluster_ids)
        colors = self.colormap(values)
        assert colors.shape == (len(values), 4)
        return colors
