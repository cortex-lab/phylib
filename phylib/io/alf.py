# -*- coding: utf-8 -*-

"""ALF dataset generation."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path
import shutil
import ast
import uuid

from tqdm import tqdm
import numpy as np

from phylib.utils._misc import _read_tsv_simple, _write_tsv_simple, ensure_dir_exists
from phylib.io.array import _spikes_per_cluster, _unique
from phylib.io.model import load_model

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# File utils
#------------------------------------------------------------------------------

NSAMPLE_WAVEFORMS = 500  # number of waveforrms sampled out of the raw data

_FILE_RENAMES = [  # file_in, file_out, squeeze (bool to squeeze vector from matlab in npy)
    ('params.py', 'params.py', None),
    ('cluster_metrics.csv', 'clusters.metrics.csv', None),
    ('spike_clusters.npy', 'spikes.clusters.npy', True),
    ('spike_templates.npy', 'spikes.templates.npy', True),
    ('channel_positions.npy', 'channels.localCoordinates.npy', False),
    ('channel_probe.npy', 'channels.probes.npy', True),
    ('cluster_probes.npy', 'clusters.probes.npy', True),
    ('cluster_shanks.npy', 'clusters.shanks.npy', True),
    ('whitening_mat.npy', '_kilosort_whitening.matrix.npy', False),
    ('_phy_spikes_subset.channels.npy', '_phy_spikes_subset.channels.npy', False),
    ('_phy_spikes_subset.spikes.npy', '_phy_spikes_subset.spikes.npy', False),
    ('_phy_spikes_subset.waveforms.npy', '_phy_spikes_subset.waveforms.npy', False),
    # ('cluster_group.tsv', 'ks2/clusters.phyAnnotation.tsv', False), # todo check indexing, add2QC
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
        self.ampfactor = model.ampfactor
        self.dir_path = Path(model.dir_path)
        self.spc = _spikes_per_cluster(model.spike_clusters)
        self.cluster_ids = _unique(self.model.spike_clusters)
        self.cluster_uuids = self.load_uuids()  # dict cluster_id => uuid

    def convert(self, out_path, force=False, label=''):
        """Convert from KS/phy format to ALF."""
        # TODO: no longer need the ampfactor parameter as it is a model attribute
        logger.info("Converting dataset to ALF in %s with ampfactor %.5f.",
                    out_path, self.ampfactor)
        self.out_path = Path(out_path)
        self.label = label
        if self.out_path.resolve() == self.dir_path.resolve():
            raise IOError("The source and target directories cannot be the same.")
        if not self.out_path.exists():
            self.out_path.mkdir()

        with tqdm(desc="Converting to ALF", total=135) as bar:
            # NOTE: this must occur BEFORE make_cluster_objects which will load subset waveforms.
            self.model.save_spikes_subset_waveforms(NSAMPLE_WAVEFORMS)
            bar.update(50)

            self.make_cluster_objects()
            bar.update(10)

            self.make_channel_objects()
            bar.update(5)

            self.make_template_and_spikes_objects()
            bar.update(30)

            self.make_depths()
            bar.update(20)

            self.rm_files()
            bar.update(10)

            self.copy_files(force=force)
            bar.update(10)

            self.rename_with_label()
            self.update_params()

        # Return the TemplateModel of the converted ALF dataset if the params.py file exists.
        params_path = self.out_path / 'params.py'
        if params_path.exists():
            return load_model(params_path)

    def load_uuids(self):
        """Load or create cluster UUIDs."""
        path = self.dir_path / 'clusters.uuids.csv'
        uuids = {}
        if path.exists():
            uuids.update(_load(path))
        missing = set(self.cluster_ids) - uuids.keys()
        uuids.update({cl: str(uuid.uuid4()) for cl in missing})
        assert set(uuids) == set(self.cluster_ids)
        return uuids

    def copy_files(self, force=False):
        for fn0, fn1, squeeze in _FILE_RENAMES:
            f0 = self.dir_path / fn0
            f1 = self.out_path / fn1
            _copy_if_possible(f0, f1, force=force)
            if f0.exists() and squeeze and f0.suffix == '.npy':
                h = _read_npy_header(f0)
                # ks2 outputs vectors as multidimensional arrays. If there is no distinction
                # for Matlab, there is one in Numpy
                if len(h['shape']) == 2 and h['shape'][-1] == 1:  # pragma: no cover
                    d = np.load(f0)
                    np.save(f1, d.squeeze())
                    continue

    def update_params(self):
        # Append ampfactor = xxx in params.py if needed.
        path = self.dir_path / 'params.py'
        assert path.exists()
        if 'ampfactor' in path.read_text():
            return
        with path.open('a') as f:
            f.write('ampfactor = %.5e' % self.ampfactor)

    def rm_files(self):
        for fn0 in FILE_DELETES:
            fn = self.dir_path.joinpath(fn0)
            if fn.exists():  # pragma: no cover
                fn.unlink()

    # File creation
    # -------------------------------------------------------------------------

    def _save_npy(self, filename, arr):
        """Save an array into a .npy file."""
        logger.debug("Save %s.", self.out_path / filename)
        np.save(self.out_path / filename, arr)

    def get_cluster_channels(self, cluster_id):
        spike_ids = self.spc[cluster_id]
        assert spike_ids.dtype == np.int64
        assert spike_ids.dtype == np.int64
        st = self.model.spike_templates[spike_ids]
        template_ids, counts = np.unique(st, return_counts=True)
        ind = np.argmax(counts)
        template_id = template_ids[ind]
        channel_ids = self.model.get_template(template_id).channel_ids[:self.model.n_channels_loc]
        return channel_ids

    def get_cluster_waveform(self, cluster_id, channel_ids):
        spike_ids = self.spc[cluster_id]
        # NOTE: keep the subset spikes so that we load the spike waveforms from the
        # already-extracted _phy_subset.waveforms.npy rather than from compressed raw data
        # which is extremely slow.
        if self.model.spike_waveforms is not None:
            assert self.model.spike_waveforms.spike_ids.dtype == np.int64
            spike_ids = np.intersect1d(spike_ids, self.model.spike_waveforms.spike_ids)
        else:
            logger.warning(
                "Loading cluster waveforms from raw data as subset spike waveforms "
                "are not available: this may be slow")
        if len(spike_ids) == 0:  # pragma: no cover
            logger.error("Empty spikes from subset waveforms for cluster %d", cluster_id)
            return
        waveforms = self.model.get_waveforms(spike_ids, channel_ids)
        if waveforms is not None:
            return waveforms.mean(axis=0)

    def make_cluster_objects(self):
        """Create clusters objects."""

        # Cluster channels and amplitudes, which may be overriden below if spike waveforms
        # or raw data are available.
        peak_channel_path = self.dir_path / 'clusters.channels.npy'
        if not peak_channel_path.exists():
            self._save_npy(peak_channel_path.name, self.model.templates_channels)

        waveform_duration_path = self.dir_path / 'clusters.peakToTrough.npy'
        if not waveform_duration_path.exists():
            self._save_npy(waveform_duration_path.name, self.model.templates_waveforms_durations)

        # group by average over cluster number
        camps = np.zeros(np.max(self.cluster_ids) - np.min(self.cluster_ids) + 1,) * np.nan
        # camps[self.cluster_ids - np.min(self.cluster_ids)] = self.model.template_amplitudes
        camps[:] = self.model.template_amplitudes
        if camps.mean() >= 1:
            logger.warning(
                "Cluster amplitudes are > 1, there might be a unit problem "
                "(check ampfactor in params.py)")
        self._save_npy('clusters.amps.npy', camps)

        # Save clusters uuids
        _write_tsv_simple(self.out_path / 'clusters.uuids.csv', "uuids", self.cluster_uuids)

        # Cluster waveforms.
        n_clusters = len(self.cluster_ids)
        nsw = self.model.n_samples_waveforms
        nc = self.model.n_channels_loc

        cluster_waveforms = np.empty((n_clusters, nsw, nc), dtype=np.float32)
        cluster_waveforms_channels = np.empty((n_clusters, nc), dtype=np.int32)

        for i, cl in enumerate(self.cluster_ids):
            channel_ids = self.get_cluster_channels(cl)
            w = self.get_cluster_waveform(cl, channel_ids)
            if w is None:
                logger.debug("Skipping the export of unavailable cluster waveforms")
                return
            ncw = w.shape[1]
            assert ncw == len(channel_ids)
            assert ncw <= nc
            assert cluster_waveforms[i, :, :ncw].shape == w.shape
            cluster_waveforms[i, :, :ncw] = w
            cluster_waveforms_channels[i, :ncw] = channel_ids

        # Save cluster waveforms.
        self._save_npy('clusters.waveforms.npy', cluster_waveforms)
        self._save_npy('clusters.waveformsChannels.npy', cluster_waveforms_channels)

        # Cluster amplitudes.
        wave_max = np.max(cluster_waveforms, axis=1) - np.min(cluster_waveforms, axis=1)
        assert wave_max.shape == (n_clusters, nc)

        cluster_amps = np.max(wave_max, axis=1)
        assert cluster_amps.shape == (n_clusters,)
        self._save_npy('clusters.amps.npy', cluster_amps)

        # Cluster channels.
        ind = np.argmax(wave_max, axis=1)
        assert ind.shape == (n_clusters,)
        assert np.all(ind < nc)
        assert cluster_waveforms_channels.shape == (n_clusters, nc)
        cluster_ch = cluster_waveforms_channels[np.arange(n_clusters), ind]
        assert cluster_ch.shape == (n_clusters,)
        assert np.all(cluster_ch >= 0)
        self._save_npy('clusters.channels.npy', cluster_ch)

    def make_channel_objects(self):
        """If there is no rawInd file, create it"""
        rawInd_path = self.dir_path / 'channels.rawInd.npy'
        rawInd = np.zeros_like(self.model.channel_probes).astype(np.int)
        channel_offset = 0
        for probe in np.unique(self.model.channel_probes):
            ind = self.model.channel_probes == probe
            rawInd[ind] = self.model.channel_mapping[ind] - channel_offset
            channel_offset += np.max(self.model.channel_mapping[ind])
        self._save_npy(rawInd_path.name, rawInd)

    def make_depths(self):
        """Make spikes.depths.npy, clusters.depths.npy."""
        channel_positions = self.model.channel_positions
        assert channel_positions.ndim == 2

        spike_clusters = self.model.spike_clusters
        assert spike_clusters.ndim == 1

        cluster_channels = np.load(self.out_path / 'clusters.channels.npy')
        assert cluster_channels.ndim == 1
        n_clusters = cluster_channels.shape[0]

        clusters_depths = channel_positions[cluster_channels, 1]
        assert clusters_depths.shape == (n_clusters,)

        if self.model.sparse_features is None:
            spikes_depths = clusters_depths[spike_clusters]
        else:
            spikes_depths = self.model.get_depths()
        self._save_npy('spikes.depths.npy', spikes_depths)
        self._save_npy('clusters.depths.npy', clusters_depths)

    def make_template_and_spikes_objects(self):
        """Creates the template waveforms sparse object
        Without manual curation, it also corresponds to clusters waveforms objects.
        """
        # We cannot just rename/copy spike_times.npy because it is in unit of samples,
        # and not seconds
        self._save_npy('spikes.times.npy', self.model.spike_times)
        self._save_npy('spikes.samples.npy', self.model.spike_samples)

        spike_amps = self.model.amplitudes
        template_amps = self.model.template_amplitudes

        # Make sure we're in volts.
        if not (np.all(spike_amps < 1) and
                np.all(template_amps < 1) and
                np.all(self.model.sparse_templates.data < 1)):
            logger.warning("Unit doesn't seem to be in volts, ampfactor may be incorrect!")

        self._save_npy('spikes.amps.npy', spike_amps)
        self._save_npy('templates.amps.npy', template_amps)

        self._save_npy('templates.waveforms.npy', self.model.sparse_templates.data)
        self._save_npy('templates.waveformsChannels.npy', self.model.sparse_templates.cols)

    def rename_with_label(self):
        """add the label as an ALF part name before the extension if any label provided"""
        if not self.label:
            return
        glob_patterns = ['channels.*', 'clusters.*', 'spikes.*', 'templates.*']
        for pattern in glob_patterns:
            for f in self.out_path.glob(pattern):
                f.rename(f.with_suffix(f'.{self.label}{f.suffix}'))
