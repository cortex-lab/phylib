# -*- coding: utf-8 -*-

"""Test fixtures."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os
from pathlib import Path
import shutil
# from textwrap import dedent

import numpy as np
from pytest import fixture

from phylib.utils._misc import write_text, write_tsv
from ..model import load_model, write_array
from phylib.io.datasets import download_test_file

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Fixtures
#------------------------------------------------------------------------------

DATASETS = {
    'template': [
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
    ],
    'ks2': [
        'ibl/ks2/params.py',
        'ibl/ks2/_spikeglx_ephysData_g0_t0.imec0.ap.bin',
        'ibl/ks2/amplitudes.npy',
        'ibl/ks2/channel_map.npy',
        'ibl/ks2/channel_positions.npy',
        'ibl/ks2/pc_feature_ind.npy',
        'ibl/ks2/pc_features.npy',
        'ibl/ks2/similar_templates.npy',
        'ibl/ks2/spike_clusters.npy',
        'ibl/ks2/spike_templates.npy',
        'ibl/ks2/spike_times.npy',
        'ibl/ks2/template_feature_ind.npy',
        'ibl/ks2/template_features.npy',
        'ibl/ks2/templates_ind.npy',
        'ibl/ks2/templates.npy',
        'ibl/ks2/whitening_mat_inv.npy',
        'ibl/ks2/whitening_mat.npy',
    ],
    'alf': [
        'ibl/alf/params.py',
        'ibl/alf/_spikeglx_ephysData_g0_t0.imec0.ap.bin',
        # 'ibl/alf/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
        # 'ibl/alf/_spikeglx_ephysData_g0_t0.imec0.ap.ch',
        'ibl/alf/channels.localCoordinates.npy',
        'ibl/alf/channels.rawInd.npy',
        'ibl/alf/clusters.amps.npy',
        'ibl/alf/clusters.channels.npy',
        'ibl/alf/clusters.depths.npy',
        # 'ibl/alf/clusters.metrics.csv',
        # 'ibl/alf/clusters.peakToTrough.npy',
        # 'ibl/alf/clusters.uuids.csv',
        # 'ibl/alf/clusters.waveformsChannels.npy',
        # 'ibl/alf/clusters.waveforms.npy',
        # 'ibl/alf/_kilosort_whitening.matrix.npy',
        'ibl/alf/_phy_spikes_subset.channels.npy',
        'ibl/alf/_phy_spikes_subset.spikes.npy',
        'ibl/alf/_phy_spikes_subset.waveforms.npy',
        'ibl/alf/spikes.amps.npy',
        'ibl/alf/spikes.clusters.npy',
        'ibl/alf/spikes.depths.npy',
        'ibl/alf/spikes.samples.npy',
        'ibl/alf/spikes.templates.npy',
        'ibl/alf/spikes.times.npy',
        # 'ibl/alf/templates.amps.npy',
        'ibl/alf/templates.waveformsChannels.npy',
        'ibl/alf/templates.waveforms.npy',
        # 'ibl/alf/whitening_mat_inv.npy',
    ],
}

DATASETS_PARAMS = ('dense', 'sparse', 'misc', 'alf', 'ks2')


def _remove(path):
    if path.exists():
        path.unlink()
        logger.debug("Removed %s.", path)


def _make_misc(tempdir):
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


def _make_extra_files(tempdir):
    logger.debug("Make extra files for mock datasets.")
    write_array(tempdir / 'spike_fail.npy', np.full(10, np.nan))  # wrong number of spikes
    write_array(tempdir / 'spike_works.npy', np.random.rand(314))
    write_array(tempdir / 'spike_randn.npy', np.random.randn(314, 2))

    # TSV file with cluster data.
    write_tsv(
        tempdir / 'cluster_Amplitude.tsv', [{'cluster_id': 1, 'Amplitude': 123.4}],
        first_field='cluster_id')

    write_tsv(
        tempdir / 'cluster_met.tsv', [
            {'cluster_id': 2, 'met1': 123.4, 'met2': 'hello world 1'},
            {'cluster_id': 3, 'met1': 5.678},
            {'cluster_id': 5, 'met2': 'hello world 2'},
        ])


class Dataset(object):
    def __init__(self, tempdir, param):
        np.random.seed(0)
        self.tempdir = Path(tempdir)
        self.param = param
        self.files = DATASETS.get(param, DATASETS['template'])
        self.params_path = tempdir / 'params.py'
        if self.param == 'sparse':
            self.files = [
                f for f in self.files if not ('_ind.' in str(f) or 'spike_ids.' in str(f))]
        self.copy()

    def path(self, name):
        return self.tempdir / name

    def copy(self):
        paths = list(map(download_test_file, self.files))
        # Copy the dataset to a temporary directory.
        for path in paths:
            to_path = self.tempdir / path.name
            to_path.parent.mkdir(exist_ok=True, parents=True)
            logger.debug("Symlinking file to %s.", to_path)
            if path.exists():
                if (path.suffix in ('.csv', '.tsv') or
                        path.name in ('spike_clusters.npy', 'spikes.clusters.npy')):
                    shutil.copy(path, to_path)
                else:
                    os.symlink(path, to_path)
            else:
                logger.warning("File %s does not exist", path)
        if self.param == 'misc':
            _make_misc(self.tempdir)
        _make_extra_files(self.tempdir)

    def create_model(self):
        self.model = load_model(self.params_path)
        self.model.param = self.param
        return self.model

    def destroy_model(self):
        self.model.close()


@fixture(scope='function', params=DATASETS_PARAMS)
def dset(tempdir, request):
    return Dataset(tempdir, request.param)


@fixture
def template_model(dset):
    yield dset.create_model()
    dset.destroy_model()
