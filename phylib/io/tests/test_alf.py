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

from phylib.utils._misc import _write_tsv
from phylib.io.alf import _FILE_RENAMES, _load, EphysAlfCreator
from phylib.io.model import TemplateModel


#------------------------------------------------------------------------------
# Fixture
#------------------------------------------------------------------------------

class Dataset(object):
    def __init__(self, tempdir):
        self.tmp_dir = tempdir
        p = Path(self.tmp_dir)
        self.ns = 100
        self.nc = 10
        self.nt = 5
        self.ncd = 1000
        np.save(p / 'spike_times.npy', .01 * np.cumsum(nr.exponential(size=self.ns)))
        np.save(p / 'spike_clusters.npy', nr.randint(low=10, high=10 + self.nt, size=self.ns))
        shutil.copy(p / 'spike_clusters.npy', p / 'spike_templates.npy')
        np.save(p / 'amplitudes.npy', nr.uniform(low=0.5, high=1.5, size=self.ns))
        np.save(p / 'channel_positions.npy', np.c_[np.arange(self.nc), np.zeros(self.nc)])
        np.save(p / 'templates.npy', np.random.normal(size=(self.nt, 50, self.nc)))
        np.save(p / 'similar_templates.npy', np.tile(np.arange(self.nt), (self.nt, 1)))
        np.save(p / 'channel_map.npy', np.c_[np.arange(self.nc)])
        _write_tsv(p / 'cluster_group.tsv', 'group', {2: 'good', 3: 'mua', 5: 'noise'})

        # Raw data
        self.dat_path = p / 'mydata.ap.bin'
        apdata = np.random.normal(size=(self.ncd, self.nc)).astype(np.int16)
        with self.dat_path.open('wb') as f:
            apdata.tofile(f)
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
    d = Dataset(tempdir)
    yield d


def test_ephys_1(dataset):
    assert dataset._load('spike_times.npy').shape == (dataset.ns,)
    assert dataset._load('spike_clusters.npy').shape == (dataset.ns,)
    assert dataset._load('amplitudes.npy').shape == (dataset.ns,)
    assert dataset._load('channel_positions.npy').shape == (dataset.nc, 2)
    assert dataset._load('templates.npy').shape == (dataset.nt, 50, dataset.nc)
    assert dataset._load('channel_map.npy').shape == (dataset.nc, 1)
    assert len(dataset._load('cluster_group.tsv')) == 3

    assert dataset._load('mydata.ap.bin').shape == (1000 * dataset.nc,)
    assert dataset._load('mydata.lf.bin').shape == (1000 * dataset.nc,)


def test_creator(dataset):
    path = Path(dataset.tmp_dir)
    out_path = path / 'alf'

    model = TemplateModel(
        dataset.dat_path, sample_rate=2000, n_channels_dat=dataset.ncd)

    c = EphysAlfCreator(model)
    with raises(IOError):
        c.convert(dataset.tmp_dir)
    c.convert(out_path)

    # Check that the raw data has been renamed.
    assert (out_path / 'ephys.raw.bin').exists()
    assert (out_path / 'lfp.raw.bin').exists()

    # Check all renames.
    for old, new in _FILE_RENAMES:
        if (path / old).exists():
            assert (out_path / new).exists()
