# -*- coding: utf-8 -*-

"""Template model providing data access functions.

The template model is based on a loader that loads data files from disk and put them
in the right in-memory format. The model should be as format-independent as possible.

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
# import os.path as op
# from pathlib import Path

import numpy as np

from .array import _spikes_per_cluster, _index_of
from .loader import from_sparse as from_sparse_
from phylib.utils import Bunch

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

def from_sparse(data, cols, channel_ids):
    # HACK: transpose so that the channels are in the last dimension
    if data.ndim == 3:
        data = np.transpose(data, (0, 2, 1))
    dense = from_sparse_(data, cols, channel_ids)
    if data.ndim == 3:
        dense = np.transpose(dense, (0, 2, 1))
    return dense


def from_sparse_bidim(data=None, rows=None, cols=None, channel_ids=None, spike_ids=None):
    assert spike_ids is not None
    if rows is not None:
        spike_ids = np.intersect1d(spike_ids, rows)
    if not len(spike_ids):
        return
    dense = from_sparse(data, cols, channel_ids)
    return Bunch(data=dense, rows=spike_ids, cols=channel_ids)


#------------------------------------------------------------------------------
# Template model
#------------------------------------------------------------------------------

class TModel:
    """Expose a dataset represented as a set of files on disk.

    Methods requesting cluster information do not reflect clustering changes during manual sorting:
    this is the responsability of the Clustering, ClusterMetadata, and Supervisor objects.

    """

    def __init__(self, loader):
        self.loader = loader
        self.spc = _spikes_per_cluster(self.loader.spike_clusters)

        self.n_spikes = len(self.loader.spike_clusters)
        self.n_channels = len(self.loader.channel_map)

        self.template_ids = np.unique(self.loader.spike_templates)
        self.n_templates = len(self.template_ids)

        self.cluster_ids = np.unique(self.loader.spike_clusters)
        self.n_clusters = len(self.cluster_ids)

    # Templates and clusters
    # ----------------------

    def template(self, tid):
        # TODO: reorder channels by decreasing template amplitude? or not needed?
        # w = self.loader.templates.data[tid]
        # amps = w.max(axis=0) - w.min(axis=0)
        return Bunch(
            waveforms=self.loader.templates.data[tid],
            channel_ids=self.loader.templates.cols[tid],
            amplitude=self.loader.template_amps[tid],
            n_spikes=len(self.spc[tid]),
        )

    def cluster(self, cid, return_waveforms=True):
        # waveforms, channel_ids, spike_count, amplitude, template_ids, template_counts
        spike_ids = self.spc[cid]
        # Templates included in the cluster
        st = self.loader.spike_templates[spike_ids]
        # Histogram of the templates for that cluster
        template_ids, counts = np.unique(st, return_counts=True)
        # Reorder by decreasing contribution of the templates
        idx = np.argsort(counts)[::-1]
        template_ids = template_ids[idx]
        counts = counts[idx]
        # Best template
        template_id = template_ids[0]
        # Best channels
        largest_template = self.template(template_id)
        channel_ids = largest_template.channel_ids
        # Amplitude
        amplitude = np.average(self.loader.template_amps[template_ids], weights=counts)
        out = Bunch(
            channel_ids=channel_ids,
            amplitude=amplitude,
            n_spikes=len(spike_ids),
            template_ids=template_ids,
            template_counts=counts,
        )
        # Waveforms
        if return_waveforms:
            out.waveforms = from_sparse(
                self.loader.templates.data[template_ids],
                self.loader.templates.cols[template_ids],
                channel_ids)
            out.waveforms = np.average(out.waveforms, weights=counts, axis=0)
            assert out.waveforms.ndim == 2
        return out

    # Spikes
    # ------
    #
    # These methods return sparse arrays as Bunch(data, cols, rows)
    # Both (optional) cols and rows are 1D arrays

    def _idx(self, x, cid, spike_ids=None):
        spike_ids = spike_ids if spike_ids is not None else self.spc.get(cid, None)
        if spike_ids is not None:
            return x[spike_ids]
        else:
            return x

    def spike_amps(self, cid):
        return self._idx(self.loader.spike_amps, cid)

    def spike_depths(self, cid):
        return self._idx(self.loader.spike_depths, cid)

    def spike_waveforms(self, cid, spike_ids=None):
        if not getattr(self.loader, 'spike_waveforms', None):
            return
        if spike_ids is None:
            spike_ids = self.spc[cid]
        sw = self.loader.spike_waveforms
        channel_ids = self.cluster(cid, return_waveforms=False)
        return from_sparse_bidim(
            data=sw.data, rows=sw.rows, cols=sw.cols,  # 3D sparse array
            channel_ids=channel_ids, spike_ids=spike_ids)  # the columns and rows we want to get

    def mean_spike_waveforms(self, cid):
        w = self.spike_waveforms(cid)
        if w is not None:
            return w.mean(axis=0)

    def _features(self, obj, cid):
        spike_ids = self.spc[cid]
        channel_ids = self.cluster(cid, return_waveforms=False).channel_ids
        # 3D sparse array
        return from_sparse_bidim(
            data=obj.data,
            rows=obj.rows,
            cols=obj.cols[_index_of(self.loader.spike_templates, self.template_ids)],
            channel_ids=channel_ids, spike_ids=spike_ids)  # the columns and rows we want to get

    def pc_features(self, cid):
        return self._features(self.loader.features, cid)

    def template_features(self, cid):
        return self._features(self.loader.template_features, cid)

    # Properties
    # ----------

    @property
    def channel_map(self):
        return self.loader.channel_map

    @property
    def channel_positions(self):
        return self.loader.channel_positions

    @property
    def channel_shanks(self):
        return self.loader.channel_shanks

    @property
    def channel_probes(self):
        return self.loader.channel_probes
