# -*- coding: utf-8 -*-

"""Probe merging."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
from pathlib import Path

from tqdm import tqdm
import numpy as np
from scipy.linalg import block_diag

from phylib.utils._misc import (
    _read_tsv_simple, _write_tsv_simple, write_tsv, read_python, write_python)
from phylib.io.model import load_model

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Merge utils
#------------------------------------------------------------------------------

def _concat(arrs, axis=0, dtype=None):
    dtype = dtype or arrs[0].dtype
    return np.concatenate(arrs).astype(dtype)


def _load_multiple_spike_times(*spike_times_l):
    """Load multiple spike_times arrays and merge them into a single one."""
    # We concatenate all spike times arrays.
    spike_times_concat = _concat(spike_times_l)
    # We sort by increasing time.
    spike_order = np.argsort(spike_times_concat, kind='stable')
    spike_times_ordered = spike_times_concat[spike_order]
    assert np.all(np.diff(spike_times_ordered) >= 0)
    # We return the ordered spike times, and the reordering array.
    return spike_times_ordered, spike_order


def _load_multiple_spike_arrays(*spike_array_l, spike_order=None):
    """Load multiple spike-dependent arrays and concatenate them along the first dimension.
    Keep the spike time ordering.
    """
    assert spike_order is not None
    spike_array_concat = _concat(spike_array_l, axis=0)
    assert spike_array_concat.shape[0] == spike_order.shape[0]
    return spike_array_concat[spike_order]


def _load_multiple_files(fn, subdirs):
    """Load the same filename in the different subdirectories."""
    # Warning: squeeze may fail in degenerate cases.
    return [np.load(str(subdir / fn)).squeeze() for subdir in subdirs]


#------------------------------------------------------------------------------
# Main Merger class
#------------------------------------------------------------------------------

class Merger(object):
    """Merge spike-sorted data from different probes and output datasets to disk.

    Constructor
    -----------

    subdirs : list
        List of paths to the probe directories.
    out_dir : str or Path
        Output directory for the merged spike-sorted data.
    probe_info : list
        For each probe, a dictionary with the following fields:
            * `label`: a string with the probe label (defaults to the probe folder name)
            * any other field, such as `model`, `serial_number`, etc.

        All fields will be saved in `probes.description.tsv`.

    """
    def __init__(self, subdirs, out_dir, probe_info=None):
        assert subdirs
        self.subdirs = [Path(subdir) for subdir in subdirs]

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Default probe info if not provided: the label is the probe folder name.
        self.probe_info = probe_info or [{'label': subdir.parts[-1]} for subdir in self.subdirs]
        assert len(self.probe_info) == len(self.subdirs)

    def _save(self, name, arr):
        """Save a npy array in the output directory."""
        logger.debug("Saving %s %s %s.", name, arr.dtype, arr.shape)
        np.save(self.out_dir / name, arr)

    def write_params(self):
        """Write a params.py for the merged dataset."""
        params_l = [read_python(subdir / 'params.py') for subdir in self.subdirs]
        n_channels_dat = sum(params['n_channels_dat'] for params in params_l)

        params_merged = params_l[0]
        params_merged['dat_path'] = []
        params_merged['n_channels_dat'] = n_channels_dat

        write_python(self.out_dir / 'params.py', params_merged)

    def write_probe_desc(self):
        """Write the probe description in a TSV file."""
        write_tsv(self.out_dir / 'probes.description.tsv', self.probe_info, first_field='label')

    def write_spike_times(self):
        """Write the merged spike times, and register self.spike_order with the reordering
        of the spikes."""
        spike_times_l = _load_multiple_files('spike_times.npy', self.subdirs)
        spike_times, self.spike_order = _load_multiple_spike_times(*spike_times_l)
        self._save('spike_times.npy', spike_times)

    def write_spike_data(self):
        """Write spike-dependent data."""
        spike_data = [
            'amplitudes.npy',
            'spike_templates.npy',
            # 'pc_features.npy',
            # 'template_features.npy',
        ]
        for fn in spike_data:
            arrays = _load_multiple_files(fn, self.subdirs)
            concat = _load_multiple_spike_arrays(*arrays, spike_order=self.spike_order)
            self._save(fn, concat)

    def write_spike_clusters(self):
        """Write the merged spike clusters, and register self.cluster_offsets.
           Write the merged spike templates, and register self.template_offsets.
        """
        spike_clusters_l = _load_multiple_files('spike_clusters.npy', self.subdirs)
        spike_templates_l = _load_multiple_files('spike_templates.npy', self.subdirs)
        self.cluster_offsets = []
        self.template_offsets = []
        cluster_probes_l = []
        coffset = 0
        toffset = 0
        for i, (subdir, sc, st) in enumerate(
                zip(self.subdirs, spike_clusters_l, spike_templates_l)):
            n_clu = np.max(sc) + 1
            n_tmp = np.max(st) + 1
            sc += coffset
            st += toffset
            self.cluster_offsets.append(coffset)
            self.template_offsets.append(toffset)
            cluster_probes_l.append(i * np.ones(n_clu, dtype=np.int32))
            coffset += n_clu
            toffset += n_tmp
        spike_clusters = _load_multiple_spike_arrays(
            *spike_clusters_l, spike_order=self.spike_order)
        spike_templates = _load_multiple_spike_arrays(
            *spike_templates_l, spike_order=self.spike_order)
        cluster_probes = _concat(cluster_probes_l)
        assert np.max(spike_clusters) + 1 == cluster_probes.size
        self._save('spike_clusters.npy', spike_clusters)
        self._save('spike_templates.npy', spike_templates)
        self._save('cluster_probes.npy', cluster_probes)

    def write_cluster_data(self):
        """We load all cluster metadata from TSV files, renumber the clusters,
        merge the dictionaries, and save in a new merged TSV file. """

        cluster_data = [
            'cluster_Amplitude.tsv',
            'cluster_ContamPct.tsv',
            'cluster_KSLabel.tsv'
        ]

        for fn in cluster_data:
            metadata = {}
            for subdir, offset in zip(self.subdirs, self.cluster_offsets):
                try:
                    field_name, metadata_loc = _read_tsv_simple(subdir / fn)
                except ValueError:
                    # Skipping non-existing file.
                    continue
                for k, v in metadata_loc.items():
                    metadata[k + offset] = v
            if metadata:
                _write_tsv_simple(self.out_dir / fn, field_name, metadata)

    def write_channel_data(self):
        """Write channel-dependent data, and register self.channel_offsets."""
        self.channel_offsets = []
        channel_probes = []
        channel_maps_l = _load_multiple_files('channel_map.npy', self.subdirs)
        # TODO if needed: channel_shanks.npy
        offset = 0
        for ind, array in enumerate(channel_maps_l):
            array += offset
            self.channel_offsets.append(offset)
            offset = array.max()
            channel_probes.append(array * 0 + ind)
        channel_maps = _concat(channel_maps_l, axis=0)
        channel_probes = _concat(channel_probes, axis=0)
        self._save('channel_map.npy', channel_maps)
        self._save('channel_probe.npy', channel_probes)

    def write_channel_positions(self):
        """Write the channel positions."""
        channel_positions_l = _load_multiple_files('channel_positions.npy', self.subdirs)
        x_offset = 0.
        for array in channel_positions_l:
            array[:, 0] += x_offset
            x_offset = 2. * array[:, 0].max() - array[:, 0].min()
        channel_positions = _concat(channel_positions_l, axis=0)
        self._save('channel_positions.npy', channel_positions)

    def write_templates(self):
        """Write the templates (only dense format for now)."""
        # TODO: write the templates array in sparse format.

        path = self.out_dir / 'templates.npy'

        templates_l = _load_multiple_files('templates.npy', self.subdirs)

        # Determine the templates array shape.
        n_templates = sum(tmp.shape[0] for tmp in templates_l)
        n_samples = templates_l[0].shape[1]  # assuming all have the same number of samples
        assert np.all(np.array([templates_i.shape[1] for templates_i in templates_l]) == n_samples)

        n_channels = sum(tmp.shape[2] for tmp in templates_l)
        shape = (n_templates, n_samples, n_channels)

        np.save(path, np.empty(shape, dtype=templates_l[0].dtype))
        offset = 0
        with open(path, 'r+b') as fid:
            fid.seek(8)
            offset = int.from_bytes(fid.read(2), byteorder='little')
            fid.seek(offset, 1)
            for i in range(len(self.subdirs)):
                j0 = templates_l[i - 1].shape[2] if i > 0 else 0
                j1 = j0 + templates_l[i].shape[2]
                for it in np.arange(templates_l[i].shape[0]):
                    one_template = np.zeros((n_samples, n_channels), dtype=templates_l[0].dtype)
                    one_template[:, j0:j1] = templates_l[i][it, :]
                    fid.write(one_template.tobytes())

    def write_template_data(self):
        template_data = [
            # 'templates_ind.npy',  # HACK: do not copy this array (which is trivial with 0 1 2 3..
            # on each row),
            # the templates.npy file is really dense in KS2 and should stay this way
            'pc_feature_ind.npy',
            'template_feature_ind.npy',
        ]

        for fn in template_data:
            arrays = _load_multiple_files(fn, self.subdirs)
            # For ind arrays, we need to take into account the channel offset.
            for array, offset in zip(arrays, self.channel_offsets):
                array += offset
            concat = _concat(arrays, axis=0).astype(np.uint32)
            self._save(fn, concat)

    def write_misc(self):
        """Write misc merged data.

        Similar templates: we make a block diagonal matrix from the n_templates * n_templates
        matrices, assuming no similarity between templates from different probes.

        Whitening matrix: same thing, except that the matrices are n_channels * n_channels.

        """
        diag_data = [
            'similar_templates.npy',
            'whitening_mat.npy',
            'whitening_mat_inv.npy',
        ]
        for fn in diag_data:
            try:
                concat = block_diag(*_load_multiple_files(fn, self.subdirs))
            except FileNotFoundError:
                logger.debug("File %s not found, skipping.", fn)
                continue
            self._save(fn, concat)

    def merge(self):
        """Merge the probes data and return a TemplateModel instance of the merged data."""

        with tqdm(desc="Merging", total=100) as bar:
            self.write_params()
            self.write_probe_desc()
            bar.update(10)
            self.write_spike_times()
            bar.update(10)
            self.write_spike_data()
            bar.update(10)
            self.write_spike_clusters()
            bar.update(10)
            self.write_cluster_data()
            bar.update(10)
            self.write_channel_data()
            bar.update(10)
            self.write_channel_positions()
            bar.update(10)
            self.write_templates()
            bar.update(10)
            self.write_template_data()
            bar.update(10)
            self.write_misc()
            bar.update(10)

        return load_model(self.out_dir / 'params.py')
