# -*- coding: utf-8 -*-

"""ALF dataset generation."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import shutil
import ast

from tqdm import tqdm
import numpy as np

from phylib.utils._misc import _read_tsv_simple, ensure_dir_exists
from phylib.io.array import _spikes_per_cluster, select_spikes, _unique, grouped_mean, _index_of
from phylib.io.model import load_model

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# File utils
#------------------------------------------------------------------------------


# ## TODO

# probes.insertion
# probes.sitePositions
# probes.rawFilename
# channels.brainLocation

_FILE_RENAMES = [  # file_in, file_out, squeeze (bool to squeeze vector from matlab in npy)
    ('params.py', 'params.py', None),
    ('spike_clusters.npy', 'spikes.clusters.npy', True),
    ('amplitudes.npy', 'spikes.amps.npy', True),
    ('channel_positions.npy', 'channels.sitePositions.npy', False),
    ('templates.npy', 'clusters.templateWaveforms.npy', False),
    ('channel_map.npy', 'channels.rawRow.npy', True),
    ('channel_map.npy', 'channels.rawRow.npy', True),
    ('channel_probe.npy', 'channels.probes.npy', True),
    ('cluster_probes.npy', 'clusters.probes.npy', True),
    ('cluster_shanks.npy', 'clusters.shanks.npy', True),

    # ('probes.description.txt', 'probes.description.txt', False),
    # ('spike_templates.npy', 'ks2/spikes.clusters.npy', True),
    # ('cluster_ContamPct.tsv', 'ks2/clusters.ContamPct.tsv', False),
    # ('cluster_group.tsv', 'ks2/clusters.phyAnnotation.tsv', False),
    # ('cluster_KSLabel.tsv', 'ks2/clusters.group.tsv', False),
]

FILE_DELETES = [
    'temp_wh.dat',  # potentially large file that will clog the servers
]


def _read_npy_header(filename):
    d = {}
    with open(filename, 'rb') as fid:
        d['magic_string'] = fid.read(6)
        d['version'] = fid.read(2)
        d['len'] = int.from_bytes(fid.read(2), byteorder='little')
        d = {**d, **ast.literal_eval(fid.read(d['len']).decode())}
    return d


def _create_if_possible(path, new_path, force=False):
    """Prepare the copy/move/symlink of a file, by making sure the source exists
    while the destination does not."""
    if not Path(path).exists():  # pragma: no cover
        logger.warning("Path %s does not exist, skipping.", path)
        return False
    if Path(new_path).exists() and not force:  # pragma: no cover
        logger.warning("Path %s already exists, skipping.", new_path)
        return False
    ensure_dir_exists(new_path.parent)
    return True


def _copy_if_possible(path, new_path, force=False):
    if not _create_if_possible(path, new_path, force=force):
        return False
    logger.debug("Copying %s to %s.", path, new_path)
    shutil.copy(path, new_path)
    return True


def _load(path):
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith(('.csv', '.tsv')):
        return _read_tsv_simple(path)[1]  # the function returns a tuple (field, data)
    elif path.endswith('.bin'):
        # TODO: configurable dtype
        return np.fromfile(path, np.int16)


#------------------------------------------------------------------------------
# Ephys ALF creator
#------------------------------------------------------------------------------

class EphysAlfCreator(object):
    """Class for converting a dataset in KS/phy format into ALF."""

    def __init__(self, model):
        self.model = model
        self.dir_path = Path(model.dir_path)
        self.spc = _spikes_per_cluster(model.spike_clusters)

    def convert(self, out_path, force=False):
        """Convert from KS/phy format to ALF."""
        logger.info("Converting dataset to ALF.")
        self.out_path = Path(out_path)
        if self.out_path.resolve() == self.dir_path.resolve():
            raise IOError("The source and target directories cannot be the same.")
        if not self.out_path.exists():
            self.out_path.mkdir()

        with tqdm(desc="Converting to ALF", total=60) as bar:
            self.copy_files(force=force)
            bar.update(10)
            self.make_spike_times()
            bar.update(10)
            self.make_cluster_waveforms()
            bar.update(10)
            self.make_depths()
            bar.update(10)
            self.make_mean_waveforms()
            bar.update(10)
            self.rm_files()
            bar.update(10)

        # Return the TemplateModel of the converted ALF dataset if the params.py file exists.
        params_path = self.out_path / 'params.py'
        if params_path.exists():
            return load_model(params_path)

    def copy_files(self, force=False):
        for fn0, fn1, squeeze in _FILE_RENAMES:
            f0 = self.dir_path / fn0
            f1 = self.out_path / fn1
            _copy_if_possible(f0, f1, force=force)
            if f0.exists() and squeeze and f0.suffix == '.npy':
                h = _read_npy_header(f0)
                # ks2 outputs vectors as multidimensional arrays. If there is no distinction
                # for Matlab, there is one in Numpy
                if len(h['shape']) == 2 and h['shape'][-1] == 1:
                    d = np.load(f0)
                    np.save(f1, d.squeeze())
                    continue

    def rm_files(self):
        for fn0 in FILE_DELETES:
            fn = self.dir_path.joinpath(fn0)
            if fn.exists():  # pragma: no cover
                fn.unlink()

    # File creation
    # -------------------------------------------------------------------------

    def _save_npy(self, filename, arr):
        """Save an array into a .npy file."""
        np.save(self.out_path / filename, arr)

    def make_spike_times(self):
        """We cannot just rename/copy spike_times.npy because it is in unit of
        *samples*, and not in seconds."""
        self.dir_path
        self._save_npy('spikes.times.npy', self.model.spike_times)

    def make_cluster_waveforms(self):
        """Return the channel index with the highest template amplitude, for
        every template."""
        p = self.dir_path
        tmp = self.model.sparse_templates.data

        peak_channel_path = p / 'clusters.peakChannel.npy'
        if not peak_channel_path.exists():
            # Create the cluster channels file.
            n_templates, n_samples, n_channels = tmp.shape

            # Compute the peak channels for each template.
            template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
            assert template_peak_channels.shape == (n_templates,)
            self._save_npy(peak_channel_path.name, template_peak_channels)
        else:  # pragma: no cover
            template_peak_channels = np.load(peak_channel_path)

        waveform_duration_path = p / 'clusters.waveformDuration.npy'
        if not waveform_duration_path.exists():
            # Compute the peak channel waveform for each template.
            waveforms = tmp[:, :, template_peak_channels]
            durations = waveforms.argmax(axis=1) - waveforms.argmin(axis=1)
            self._save_npy(waveform_duration_path.name, durations)

    def make_depths(self):
        """Make spikes.depths.npy, clusters.depths.npy."""
        channel_positions = self.model.channel_positions
        assert channel_positions.ndim == 2

        spike_clusters = self.model.spike_clusters
        assert spike_clusters.ndim == 1
        n_spikes = spike_clusters.shape[0]
        self.cluster_ids = _unique(self.model.spike_clusters)

        cluster_channels = np.load(self.out_path / 'clusters.peakChannel.npy')
        assert cluster_channels.ndim == 1
        n_clusters = cluster_channels.shape[0]

        clusters_depths = channel_positions[cluster_channels, 1]
        assert clusters_depths.shape == (n_clusters,)

        spike_clusters_rel = _index_of(spike_clusters, self.cluster_ids)
        assert spike_clusters_rel.max() < clusters_depths.shape[0]
        spikes_depths = clusters_depths[spike_clusters_rel]
        assert spikes_depths.shape == (n_spikes,)

        self._save_npy('spikes.depths.npy', spikes_depths)
        self._save_npy('clusters.depths.npy', clusters_depths)

    def make_mean_waveforms(self):
        """Make the mean waveforms file."""
        spike_ids = select_spikes(
            cluster_ids=self.cluster_ids,
            max_n_spikes_per_cluster=100,
            spikes_per_cluster=lambda clu: self.spc[clu],
            subset='random')
        waveforms = self.model.get_waveforms(spike_ids, np.arange(self.model.n_channels))
        try:
            mean_waveforms = grouped_mean(waveforms, self.model.spike_clusters[spike_ids])
            self._save_npy('clusters.meanWaveforms.npy', mean_waveforms)
        except IndexError as e:  # pragma: no cover
            logger.warning("Failed to create the mean waveforms file: %s.", e)
