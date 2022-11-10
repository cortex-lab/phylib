# -*- coding: utf-8 -*-

"""Test ALF dataset generation."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
from pathlib import Path
import shutil
from pytest import fixture, raises

import numpy as np
import numpy.random as nr

from phylib.utils._misc import _write_tsv_simple
from ..alf import _FILE_RENAMES, _load, EphysAlfCreator
from ..model import TemplateModel


#------------------------------------------------------------------------------
# Fixture
#------------------------------------------------------------------------------

class Dataset(object):
    def __init__(self, tempdir):
        np.random.seed(42)
        self.tmp_dir = tempdir
        p = Path(self.tmp_dir)
        self.ns = 100
        self.nsamp = 25
        self.ncmax = 42
        self.nc = 10
        self.nt = 5
        self.ncd = 1000
        np.save(p / 'spike_times.npy', .01 * np.cumsum(nr.exponential(size=self.ns)))
        np.save(p / 'spike_clusters.npy', nr.randint(low=1, high=self.nt, size=self.ns))
        shutil.copy(p / 'spike_clusters.npy', p / 'spike_templates.npy')
        np.save(p / 'amplitudes.npy', nr.uniform(low=0.5, high=1.5, size=self.ns))
        np.save(p / 'channel_positions.npy', np.c_[np.arange(self.nc), np.zeros(self.nc)])
        np.save(p / 'templates.npy', np.random.normal(size=(self.nt, 50, self.nc)))
        np.save(p / 'similar_templates.npy', np.tile(np.arange(self.nt), (self.nt, 1)))
        np.save(p / 'channel_map.npy', np.c_[np.arange(self.nc)])
        np.save(p / 'channel_probe.npy', np.zeros(self.nc))
        np.save(p / 'whitening_mat.npy', np.eye(self.nc, self.nc))
        np.save(p / '_phy_spikes_subset.channels.npy', np.zeros([self.ns, self.ncmax]))
        np.save(p / '_phy_spikes_subset.spikes.npy', np.zeros([self.ns]))
        np.save(p / '_phy_spikes_subset.waveforms.npy', np.zeros(
            [self.ns, self.nsamp, self.ncmax])
        )

        _write_tsv_simple(p / 'cluster_group.tsv', 'group', {2: 'good', 3: 'mua', 5: 'noise'})
        _write_tsv_simple(p / 'cluster_Amplitude.tsv', field_name='Amplitude',
                          data={str(n): np.random.rand() * 120 for n in np.arange(self.nt)})
        with open(p / 'probes.description.txt', 'w+') as fid:
            fid.writelines(['label\n'])

        # Raw data
        self.dat_path = p / 'rawdata.npy'
        np.save(self.dat_path, np.random.normal(size=(self.ncd, self.nc)))

        # LFP data.
        lfdata = (100 * np.random.normal(size=(1000, self.nc))).astype(np.int16)
        with (p / 'mydata.lf.bin').open('wb') as f:
            lfdata.tofile(f)

        self.files = os.listdir(self.tmp_dir)

    def _load(self, fn):
        p = Path(self.tmp_dir)
        return _load(p / fn)


@fixture
def dataset(tempdir):
    return Dataset(tempdir)


def test_ephys_1(dataset):
    assert dataset._load('spike_times.npy').shape == (dataset.ns,)
    assert dataset._load('spike_clusters.npy').shape == (dataset.ns,)
    assert dataset._load('amplitudes.npy').shape == (dataset.ns,)
    assert dataset._load('channel_positions.npy').shape == (dataset.nc, 2)
    assert dataset._load('templates.npy').shape == (dataset.nt, 50, dataset.nc)
    assert dataset._load('channel_map.npy').shape == (dataset.nc, 1)
    assert dataset._load('channel_probe.npy').shape == (dataset.nc,)
    assert len(dataset._load('cluster_group.tsv')) == 3
    assert dataset._load('rawdata.npy').shape == (1000, dataset.nc)
    assert dataset._load('mydata.lf.bin').shape == (1000 * dataset.nc,)
    assert dataset._load('whitening_mat.npy').shape == (dataset.nc, dataset.nc)
    assert dataset._load('_phy_spikes_subset.channels.npy').shape == (dataset.ns, dataset.ncmax)
    assert dataset._load('_phy_spikes_subset.spikes.npy').shape == (dataset.ns,)
    assert dataset._load('_phy_spikes_subset.waveforms.npy').shape == (
        (dataset.ns, dataset.nsamp, dataset.ncmax)
    )


def test_spike_depths(dataset):
    path = Path(dataset.tmp_dir)
    out_path = path / 'alf'

    mtemp = TemplateModel(
        dir_path=path, dat_path=dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.nc)

    # create some sparse PC features
    n_subch = int(np.round(mtemp.n_channels / 2) - 1)
    pc_features = np.zeros((mtemp.n_spikes, n_subch, 3))
    close_channels = np.meshgrid(np.ones(mtemp.n_templates), np.arange(n_subch))[1]
    chind = mtemp.templates_channels
    chind = mtemp.templates_channels + close_channels * ((chind < 5) * 2 - 1)
    pc_features_ind = chind.transpose()
    # all PCs max between first and second channel
    pc_features[:, 0, 0] = 1
    pc_features[:, 1, 0] = 0.5
    print(mtemp.templates_channels)
    print(mtemp.templates_waveforms_durations)
    # add some depth information
    mtemp.channel_positions[:, 1] = mtemp.channel_positions[:, 0] + 10
    np.save(path / 'pc_features.npy', np.swapaxes(pc_features, 2, 1))
    np.save(path / 'pc_feature_ind.npy', pc_features_ind)
    np.save(path / 'channel_positions.npy', mtemp.channel_positions)

    model = TemplateModel(
        dir_path=path, dat_path=dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.nc)

    c = EphysAlfCreator(model)
    shutil.rmtree(out_path, ignore_errors=True)
    c.convert(out_path)
    sd = np.load(next(out_path.glob('spikes.depths.npy')))
    sd_ = model.channel_positions[model.templates_channels[model.spike_templates], 1]
    assert np.all(np.abs(sd - sd_) <= 0.5)


def test_creator(dataset):
    _FILE_CREATES = (
        'spikes.times*.npy',
        'spikes.depths*.npy',
        'spikes.samples*.npy',
        'clusters.uuids*.csv',
        'clusters.amps*.npy',
        'clusters.channels*.npy',
        'clusters.depths*.npy',
        'clusters.peakToTrough*.npy',
        'clusters.waveforms*.npy',
        'clusters.waveformsChannels*.npy',
        'channels.localCoordinates*.npy',
        'channels.rawInd*.npy',
        'templates.waveforms*.npy',
        'templates.waveformsChannels*.npy',
    )
    path = Path(dataset.tmp_dir)
    out_path = path / 'alf'

    model = TemplateModel(
        dir_path=path, dat_path=dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.nc)

    c = EphysAlfCreator(model)
    with raises(IOError):
        c.convert(dataset.tmp_dir)

    def check_conversion_output():
        # Check all renames.
        for old, new, _ in _FILE_RENAMES:
            if (path / old).exists():
                pattern = f'{Path(new).stem}*{Path(new).suffix}'
                assert next(out_path.glob(pattern)).exists()

        new_files = []
        for new in _FILE_CREATES:
            f = next(out_path.glob(new))
            new_files.append(f)
            assert f.exists()

        # makes sure the output dimensions match (especially clusters which should be 4)
        cl_shape = []
        for f in new_files:
            if f.name.startswith('clusters.') and f.name.endswith('.npy'):
                cl_shape.append(np.load(f).shape[0])
            elif f.name.startswith('clusters.') and f.name.endswith('.csv'):
                with open(f) as fid:
                    cl_shape.append(len(fid.readlines()) - 1)
        sp_shape = [np.load(f).shape[0] for f in new_files if f.name.startswith('spikes.')]
        ch_shape = [np.load(f).shape[0] for f in new_files if f.name.startswith('channels.')]

        assert len(set(cl_shape)) == 1
        assert len(set(sp_shape)) == 1
        assert len(set(ch_shape)) == 1

        dur = np.load(next(out_path.glob('clusters.peakToTrough*.npy')))
        assert np.all(dur == np.array([-9.5, 3., 13., -4.5, -2.5]))

    def read_after_write():
        model = TemplateModel(dir_path=out_path, dat_path=dataset.dat_path,
                              sample_rate=2000, n_channels_dat=dataset.nc)

        np.all(model.spike_templates == c.model.spike_templates)
        np.all(model.spike_times == c.model.spike_times)
        np.all(model.spike_samples == c.model.spike_samples)

    # test a straight export, make sure we can reload the data
    shutil.rmtree(out_path, ignore_errors=True)
    c.convert(out_path)
    check_conversion_output()
    read_after_write()

    # test with a label after the attribute name
    shutil.rmtree(out_path)
    c.convert(out_path, label='probe00')
    check_conversion_output()
    read_after_write()


def test_merger(dataset):

    path = Path(dataset.tmp_dir)
    out_path = path / 'alf'

    model = TemplateModel(
        dir_path=path, dat_path=dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.nc)

    c = EphysAlfCreator(model)
    c.convert(out_path)

    model.close()

    # path.joinpath('_phy_spikes_subset.channels.npy').unlink()
    # path.joinpath('_phy_spikes_subset.waveforms.npy').unlink()
    # path.joinpath('_phy_spikes_subset.spikes.npy').unlink()

    out_path_merge = path / 'alf_merge'
    spike_clusters = dataset._load('spike_clusters.npy')
    clu, n_clu = np.unique(spike_clusters, return_counts=True)

    # merge the first two clusters
    merge_clu = clu[0:2]
    spike_clusters[np.bitwise_or(spike_clusters == clu[0],
                                 spike_clusters == clu[1])] = np.max(clu) + 1
    # split the cluster with the most spikes
    split_clu = clu[-1]
    idx = np.where(spike_clusters == split_clu)[0]
    spike_clusters[idx[0:int(n_clu[-1] / 2)]] = np.max(clu) + 2
    spike_clusters[idx[int(n_clu[-1] / 2):]] = np.max(clu) + 3

    np.save(path / 'spike_clusters.npy', spike_clusters)

    model = TemplateModel(
        dir_path=path, dat_path=dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.nc)

    c = EphysAlfCreator(model)
    c.convert(out_path_merge)

    # Test that the split are the same for the expected datasets
    clu_old = np.load(next(out_path.glob('clusters.peakToTrough.npy')))
    clu_new = np.load(next(out_path_merge.glob('clusters.peakToTrough.npy')))
    assert clu_old[split_clu] == clu_new[np.max(clu) + 2]
    assert clu_old[split_clu] == clu_new[np.max(clu) + 3]

    assert np.isnan([clu_new[split_clu]])[0]
    assert np.isnan([clu_new[merge_clu[0]]])[0]
    assert np.isnan([clu_new[merge_clu[1]]])[0]

    clu_old = np.load(next(out_path.glob('clusters.channels.npy')))
    clu_new = np.load(next(out_path_merge.glob('clusters.channels.npy')))
    assert clu_old[split_clu] == clu_new[np.max(clu) + 2]
    assert clu_old[split_clu] == clu_new[np.max(clu) + 3]
    assert clu_new[split_clu] == 0
    assert clu_new[merge_clu[0]] == 0
    assert clu_new[merge_clu[1]] == 0

    clu_old = np.load(next(out_path.glob('clusters.depths.npy')))
    clu_new = np.load(next(out_path_merge.glob('clusters.depths.npy')))
    assert clu_old[split_clu] == clu_new[np.max(clu) + 2]
    assert clu_old[split_clu] == clu_new[np.max(clu) + 3]
    assert np.isnan([clu_new[split_clu]])[0]
    assert np.isnan([clu_new[merge_clu[0]]])[0]
    assert np.isnan([clu_new[merge_clu[1]]])[0]

    clu_old = np.load(next(out_path.glob('clusters.waveformsChannels.npy')))
    clu_new = np.load(next(out_path_merge.glob('clusters.waveformsChannels.npy')))
    assert np.array_equal(clu_old[split_clu], clu_new[np.max(clu) + 2])
    assert np.array_equal(clu_old[split_clu], clu_new[np.max(clu) + 3])
