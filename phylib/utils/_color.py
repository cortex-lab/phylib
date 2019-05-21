# -*- coding: utf-8 -*-

"""Color routines."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import logging

from phylib.utils import Bunch
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
# Colormap utilities
#------------------------------------------------------------------------------

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


def _spike_colors(spike_clusters=None, masks=None, alpha=None, colormap=None):
    colormap = colormap if colormap is not None else colormaps.default
    n = len(colormap)
    if spike_clusters is not None:
        c = colormap[np.mod(spike_clusters, n), :]
    else:
        c = np.ones((masks.shape[0], 3))
    return _apply_color_masks(c, masks=masks, alpha=alpha)


def _continuous_colormap(colormap, values, vmin=None, vmax=None):
    assert colormap.shape[1] == 3
    n = colormap.shape[0]
    vmin = vmin if vmin is not None else values.min()
    vmax = vmax if vmax is not None else values.max()
    assert vmin is not None
    assert vmax is not None
    denom = vmax - vmin
    denom = denom if denom != 0 else 1
    # NOTE: clipping is necessary when a view using color selector (like the raster view)
    # is updated right after a clustering update, but before the vmax had a chance to
    # be updated.
    i = np.clip(np.round((n - 1) * (values - vmin) / denom).astype(np.int32), 0, n - 1)
    return colormap[i, :]


def _categorical_colormap(colormap, values, vmin=None, vmax=None):
    assert np.issubdtype(values.dtype, np.integer)
    assert colormap.shape[1] == 3
    n = colormap.shape[0]
    if vmin is None and vmax is None:
        # Find unique values and keep the order.
        _, idx = np.unique(values, return_index=True)
        lookup = values[np.sort(idx)]
        x = _index_of(values, lookup)
    else:
        x = values
    return colormap[x % n, :]


#------------------------------------------------------------------------------
# Colormaps
#------------------------------------------------------------------------------

# Default color map for the selected clusters.
# see https://colorcet.pyviz.org/user_guide/Categorical.html
def _make_default_colormap():
    colormap = np.array(cc.glasbey_bw_minc_20_minl_30)
    # Reorder first colors.
    colormap[[0, 1, 2, 3, 4, 5]] = colormap[[3, 0, 4, 5, 2, 1]]
    # Replace first two colors.
    colormap[0] = [0.03137, 0.5725, 0.9882]
    colormap[1] = [1.0000, 0.0078, 0.0078]
    return colormap


def _make_cluster_group_colormap():
    # Rows are sorted by increasing alphabetical order (all lowercase).
    return np.array([
        [0.75, 0.75, 0.75],  # '' (None = '' = unsorted)
        [0.5254, 0.8196, 0.42745],  # good
        [0.5, 0.5, 0.5],  # mua
        [0.4, 0.4, 0.4],  # noise
    ])


# Built-in colormaps.
colormaps = Bunch(
    default=_make_default_colormap(),
    cluster_group=_make_cluster_group_colormap(),
    categorical=np.array(cc.glasbey_bw_minc_20_minl_30),
    rainbow=np.array(cc.rainbow_bgyr_35_85_c73),
    linear=np.array(cc.linear_wyor_100_45_c55),
    diverging=np.array(cc.diverging_linear_bjy_30_90_c45),
)


def selected_cluster_color(i, alpha=1.):
    return add_alpha(tuple(colormaps.default[i % len(colormaps.default)]), alpha=alpha)


#------------------------------------------------------------------------------
# Cluster color selector
#------------------------------------------------------------------------------

def add_alpha(c, alpha=1.):
    if isinstance(c, (tuple,)):
        return c + (alpha,)
    elif isinstance(c, np.ndarray):
        assert c.ndim == 2
        assert c.shape[1] == 3
        return np.c_[c, alpha * np.ones((c.shape[0], 1))]
    raise ValueError("Unknown value given in add_alpha().")


class ClusterColorSelector(object):
    """Assign a color to clusters depending on cluster labels or metrics."""
    _color_field = 'cluster'
    _colormap = colormaps.categorical
    _categorical = True

    def __init__(
            self, cluster_meta=None, cluster_metrics=None,
            field=None, colormap=None, categorical=True, cluster_ids=None):
        self.cluster_ids = None
        self.cluster_meta = cluster_meta or None
        self.cluster_meta_fields = cluster_meta.fields if cluster_meta else ()
        self.cluster_metrics = cluster_metrics or {}
        # Used to initialize the value range for the clusters.
        assert cluster_ids is not None
        self.cluster_ids = cluster_ids
        # self._colormap = colormap if colormap is not None else self._colormap
        self.set_color_mapping(field=field, colormap=colormap, categorical=categorical)

    @property
    def state(self):
        colormap_name = None
        # Find the colormap name from the colormap array.
        for cname, arr in colormaps.items():
            if self._colormap.shape == arr.shape and np.allclose(self._colormap, arr):
                colormap_name = cname
                break
        return Bunch(
            color_field=self._color_field,
            colormap=colormap_name,
            categorical=self._categorical,
        )

    def set_state(self, state):
        self.set_color_mapping(
            field=state.color_field, colormap=state.colormap, categorical=state.categorical)

    def set_color_mapping(self, field=None, colormap=None, categorical=True):
        """Set the field used to choose the cluster colors, and the associated colormap."""
        if isinstance(colormap, str):
            colormap = colormaps[colormap]
        self._colormap = colormap if colormap is not None else self._colormap
        self._color_field = field or self._color_field
        self._categorical = categorical
        # Recompute the value range.
        self.set_cluster_ids(self.cluster_ids)

    def set_cluster_ids(self, cluster_ids):
        """Precompute the value range for all clusters."""
        self.cluster_ids = cluster_ids
        values = self.get_values(self.cluster_ids)
        if values is not None:
            self.vmin, self.vmax = values.min(), values.max()

    def map(self, values):
        # Use categorical or continuous colormap depending on the data type of the values.
        f = (_categorical_colormap
             if self._categorical and np.issubdtype(values.dtype, np.integer)
             else _continuous_colormap)
        return f(self._colormap, values, vmin=self.vmin, vmax=self.vmax)
        raise ValueError("Values is neither integer or float datatype.")

    def _get_cluster_value(self, cluster_id):
        """Return the field value for a given cluster."""
        field = self._color_field
        if field == 'cluster':
            return cluster_id
        elif field in self.cluster_meta_fields:
            return self.cluster_meta.get(field, cluster_id)
        elif field in self.cluster_metrics:
            return self.cluster_metrics.get(field, lambda cl: None)(cluster_id)
        logger.warning("The field %s is not an existing cluster label or metrics.", field)
        return 0

    def get(self, cluster_id, alpha=None):
        """Return the color of a given cluster."""
        assert self.cluster_ids is not None
        assert self._colormap is not None
        val = self._get_cluster_value(cluster_id)
        col = tuple(self.map(np.array([val]))[0].tolist())
        return add_alpha(col, alpha=alpha)

    def get_values(self, cluster_ids):
        values = [self._get_cluster_value(cluster_id) for cluster_id in cluster_ids]
        # Deal with categorical variables (strings)
        if any(isinstance(v, str) or v is None for v in values):
            # HACK: replace None by empty string to avoid error when sorting the unique values.
            values = [str(v).lower() if v is not None else '' for v in values]
            uv = sorted(set(values))
            values = [uv.index(v) for v in values]
        return np.array(values)

    def get_colors(self, cluster_ids, alpha=1.):
        """Return cluster colors."""
        values = self.get_values(cluster_ids)
        assert len(values) == len(cluster_ids)
        return add_alpha(self.map(values), alpha=alpha)
