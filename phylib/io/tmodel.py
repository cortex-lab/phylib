# -*- coding: utf-8 -*-

"""Template model providing data access functions.

The template model is based on a loader that loads data files from disk and put them
in the right in-memory format. The model should be as format-independent as possible.

"""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import logging
import os.path as op
from pathlib import Path

import numpy as np

from .array import _spikes_per_cluster

logger = logging.getLogger(__name__)


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

        self.template_ids = np.unique(self.loader.spike_templates)
        self.n_templates = len(self.template_ids)
        # TODO: template sizes as a n_templates-long array

        self.cluster_ids = np.unique(self.loader.spike_clusters)
        self.n_clusters = len(self.cluster_ids)
        # TODO: clusters sizes as a n_clusters-long array

    # Templates and clusters
    # ----------------------

    def template(self, tid):
        # waveforms, channel_ids, spike_count, amplitude
        pass

    def cluster(self, cid):
        # waveforms, channel_ids, spike_count, amplitude, template_ids, template_counts
        pass

    # Spikes
    # ------
    #
    # These methods return sparse arrays as Bunch(data, cols, rows)
    # cols and rows may both be 1D

    def _idx(self, x, cid, spike_ids=None):
        spike_ids = spike_ids if spike_ids is not None else self.spc.get(cid, None)
        if spike_ids is not None:
            return x[spike_ids]

    def spike_waveforms(self, cid, spike_ids=None):
        pass

    def mean_spike_waveforms(self, cid):
        w = self.spike_waveforms(cid)
        return w.mean(axis=0)

    def pc_features(self, cid):
        pass

    def template_features(self, cid):
        pass

    def spike_amps(self, cid):
        return self._idx(self.loader.spike_amps, cid)

    def spike_depths(self, cid):
        return self._idx(self.loader.spike_depths, cid)

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
