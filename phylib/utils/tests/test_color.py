# -*- coding: utf-8 -*-

"""Test colors."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import colorcet as cc
import numpy as np

from .._color import (_is_bright, _random_bright_color,
                      _colormap, _spike_colors,
                      _continuous_colormap, _categorical_colormap,
                      categorical, rainbow, diverging,
                      ClusterColorSelector, get_colormap,
                      )


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

def test_random_color():
    for _ in range(10):
        assert _is_bright(_random_bright_color())


def test_cluster_colormap():
    assert len(_colormap(0)) == 3
    assert len(_colormap(1000)) == 3
    assert len(_colormap(0, 0.5)) == 4

    assert _spike_colors([0, 1, 10, 1000]).shape == (4, 4)
    assert _spike_colors([0, 1, 10, 1000],
                         alpha=1.).shape == (4, 4)
    assert _spike_colors([0, 1, 10, 1000],
                         masks=np.linspace(0., 1., 4)).shape == (4, 4)
    assert _spike_colors(masks=np.linspace(0., 1., 4)).shape == (4, 4)


def test_colormaps():
    colormap = np.array(cc.glasbey_bw_minc_20_minl_30)
    values = np.random.randint(10, 20, size=100)
    colors = _categorical_colormap(colormap, values)
    assert colors.shape == (100, 4)

    colormap = np.array(cc.rainbow_bgyr_35_85_c73)
    values = np.linspace(0, 1, 100)
    colors = _continuous_colormap(colormap, values)
    assert colors.shape == (100, 4)


def test_cluster_color_selector():
    cluster_labels = {'label': {1: 10, 2: 20, 3: 30}}
    cluster_metrics = {'quality': lambda c: c * .1}
    cluster_ids = [1, 2, 3]
    c = ClusterColorSelector(
        cluster_labels=cluster_labels,
        cluster_metrics=cluster_metrics,
        cluster_ids=cluster_ids,
    )

    assert len(c.get(1, alpha=.5)) == 4

    c.set_color_field(field='label', colormap=get_colormap('linear'))
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)

    c.set_color_field(field='quality', colormap=get_colormap(rainbow))
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)

    c.set_color_field(field='cluster', colormap=categorical)
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)

    c.set_color_field(field='nonexisting', colormap=diverging)
    colors = c.get_colors(cluster_ids)
    assert colors.shape == (3, 4)
