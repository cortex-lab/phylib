# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import shutil
from textwrap import dedent

import numpy as np
from pytest import fixture

from phylib.utils._misc import write_text, write_tsv
from ..model import load_model, write_array
from phylib.io.datasets import download_test_file
from phylib.io.mock import (
    artificial_spike_samples,
    artificial_spike_clusters,
    artificial_waveforms,
    artificial_traces,
)

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
    np.random.seed(0)

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
            st = np.load(tempdir / 'spike_times.npy').squeeze()
            st_r = st + np.random.randint(low=-20000, high=+20000, size=st.size)
            assert st_r.shape == st.shape
            # Reordered spikes.
            np.save(tempdir / 'spike_times_reordered.npy', st_r)
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

    write_tsv(
        tempdir / 'cluster_metrics.tsv', [
            {'cluster_id': 2, 'met1': 123.4, 'met2': 'hello world 1'},
            {'cluster_id': 3, 'met1': 5.678},
            {'cluster_id': 5, 'met2': 'hello world 2'},
        ])

    template_path = tempdir / paths[0].name
    return template_path


def _make_mock_dataset(tempdir):
    root = tempdir / 'mock'
    root.mkdir(parents=True, exist_ok=True)

    def _save(path, arr):
        np.save(root / path, arr)

    # Params.
    n_channels = 164
    n_channels_loc = 24
    n_spikes = 4_000
    n_clusters = 72
    n_samples_waveforms = 62
    sample_rate = 5e3

    n_templates = n_clusters

    # Channel positions.
    channel_positions = np.random.normal(size=(n_channels, 2))
    _save('channel_positions.npy', channel_positions)

    channel_mapping = np.arange(0, n_channels).astype(np.int32)
    _save("channel_map.npy", channel_mapping)

    # Spike times.
    # for simplicity, assume all spikes are complete on the raw data
    spike_samples = n_samples_waveforms + artificial_spike_samples(n_spikes, max_isi=20)
    # spike_times = spike_samples / sample_rate
    # WARNING: not that the KS2 file format uses "spike_times.npy" for spike SAMPLES in integers!
    _save('spike_times.npy', spike_samples.reshape((-1, 1)))

    # Spike amplitudes
    amplitudes = np.random.normal(size=n_spikes, loc=1, scale=.1)
    _save('amplitudes.npy', amplitudes.reshape((-1, 1)))

    # Spike templates.
    spike_templates = artificial_spike_clusters(n_spikes, n_clusters)
    _save('spike_templates.npy', spike_templates.astype(np.int64))

    # Template waveforms.
    templates = artificial_waveforms(n_templates, n_samples_waveforms, n_channels)
    # NOTE: simulate "fake sparse" output by KS2. The template channels is trivial and does
    # not contain localization information, which will have to be recovered by the
    # TemplateModel.
    template_channels = np.zeros((n_templates, n_channels), dtype=np.int64)
    # Space localization.
    ch = np.arange(n_channels)
    amp = np.exp(-ch / n_channels_loc)
    for i in range(n_templates):
        perm = np.random.permutation(ch)
        templates[i, :, perm] *= amp[:, np.newaxis]
        template_channels[i, :] = ch

    # Raw data.
    # for simplicity, assume all spikes are complete on the raw data
    duration = spike_samples[-1] + n_samples_waveforms
    traces = 1e-3 * artificial_traces(duration, n_channels)
    factor = 50000 / traces.max()
    (factor * traces).astype(np.int16).tofile(root / 'raw.dat')

    # NOTE: need to take the factor into account in the templates
    templates *= 1e-3 * factor
    _save('templates.npy', templates.astype(np.float32))
    _save('templates_ind.npy', template_channels)

    write_text(root / 'params.py', dedent(f'''
        dat_path = 'raw.dat'
        n_channels_dat = {n_channels}
        dtype = 'int16'
        offset = 0
        sample_rate = {sample_rate}
        hp_filtered = False
        ampfactor = 1.0 / %.5e
    ''' % factor))
    return root / 'params.py'


@fixture(scope='function', params=('dense', 'sparse', 'misc', 'mock'))
def template_path_full(tempdir, request):
    if request.param != 'mock':
        return _make_dataset(tempdir, request.param)
    else:
        return _make_mock_dataset(tempdir)


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
