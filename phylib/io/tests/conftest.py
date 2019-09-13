# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import shutil

import numpy as np
from pytest import fixture

from phylib.utils._misc import write_text, write_tsv
from ..model import load_model, write_array
from phylib.io.datasets import download_test_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

_FILES = [
    'template/params.py',
    'template/sim_binary.dat',
    'template/spike_times.npy',
    'template/spike_templates.npy',
    'template/spike_clusters.npy',
    'template/amplitudes.npy',

    'template/cluster_group.tsv',

    'template/channel_map.npy',
    'template/channel_positions.npy',
    'template/channel_shanks.npy',

    'template/similar_templates.npy',
    'template/whitening_mat.npy',

    'template/templates.npy',
    'template/template_ind.npy',

    'template/pc_features.npy',
    'template/pc_feature_ind.npy',
    'template/pc_feature_spike_ids.npy',

    'template/template_features.npy',
    'template/template_feature_ind.npy',
    'template/template_feature_spike_ids.npy',
]


def _remove(path):
    if path.exists():
        path.unlink()
        logger.debug("Removed %s.", path)


def _make_dataset(tempdir, param='dense', has_spike_attributes=True):
    # Download the dataset.
    paths = list(map(download_test_file, _FILES))
    # Copy the dataset to a temporary directory.
    for path in paths:
        to_path = tempdir / path.name
        # Skip sparse arrays if is_sparse is False.
        if param == 'sparse' and ('_ind.' in str(to_path) or 'spike_ids.' in str(to_path)):
            continue
        logger.debug("Copying file to %s.", to_path)
        shutil.copy(path, to_path)

    # Some changes to files if 'misc' fixture parameter.
    if param == 'misc':
        # Remove spike_clusters and recreate it from spike_templates.
        _remove(tempdir / 'spike_clusters.npy')
        # Replace spike_times.npy, in samples, by spikes.times.npy, in seconds.
        if (tempdir / 'spike_times.npy').exists():
            st = np.load(tempdir / 'spike_times.npy')
            np.save(tempdir / 'spikes.times.npy', st / 25000.)  # sample rate
            _remove(tempdir / 'spike_times.npy')
        # Buggy TSV file should not cause a crash.
        write_text(tempdir / 'error.tsv', '')
        # Remove some non-necessary files.
        _remove(tempdir / 'template_features.npy')
        _remove(tempdir / 'pc_features.npy')
        _remove(tempdir / 'channel_probes.npy')
        _remove(tempdir / 'channel_shanks.npy')
        _remove(tempdir / 'amplitudes.npy')
        _remove(tempdir / 'whitening_mat.npy')
        _remove(tempdir / 'whitening_mat_inv.npy')
        _remove(tempdir / 'sim_binary.dat')

    # Spike attributes.
    if has_spike_attributes:
        write_array(tempdir / 'spike_fail.npy', np.full(10, np.nan))  # wrong number of spikes
        write_array(tempdir / 'spike_works.npy', np.random.rand(314))
        write_array(tempdir / 'spike_randn.npy', np.random.randn(314, 2))

    # TSV file with cluster data.
    write_tsv(
        tempdir / 'cluster_Amplitude.tsv', [{'cluster_id': 1, 'Amplitude': 123.4}],
        first_field='cluster_id')

    template_path = tempdir / paths[0].name
    return template_path


@fixture(scope='function', params=('dense', 'sparse', 'misc'))
def template_path_full(tempdir, request):
    return _make_dataset(tempdir, request.param)


@fixture(scope='function')
def template_path(tempdir, request):
    return _make_dataset(tempdir, param='dense', has_spike_attributes=False)


@fixture
def template_model_full(template_path_full):
    model = load_model(template_path_full)
    yield model
    model.close()


@fixture
def template_model(template_path):
    model = load_model(template_path)
    yield model
    model.close()
