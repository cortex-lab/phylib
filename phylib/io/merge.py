from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from phylib.utils._misc import _read_tsv_simple, _write_tsv_simple
from phylib.io import model


def _load_multiple_spike_times(*spike_times_l):
    """Load multiple spike_times arrays into a single one."""
    # We concatenate all spike times arrays.
    spike_times_concat = np.concatenate(spike_times_l)
    # We sort by increasing time.
    spike_order = np.argsort(spike_times_concat)
    spike_times_ordered = spike_times_concat[spike_order]
    assert np.all(np.diff(spike_times_ordered) >= 0)
    # We return the ordered spike times, and the reordering array.
    return spike_times_ordered, spike_order


def _load_multiple_spike_arrays(*spike_array_l, spike_order=None):
    """Load multiple spike-dependent arrays and concatenate them along the first dimension.
    Keep the spike time ordering.
    """
    assert spike_order is not None
    spike_array_concat = np.concatenate(spike_array_l, axis=0)
    assert spike_array_concat.shape[0] == spike_order.shape[0]
    return spike_array_concat[spike_order]


def _load_multiple_files(fn, subdirs):
    """Load the same filename in the different subdirectories."""
    # Warning: squeeze may fail in degenerate cases.
    return [np.load(str(subdir / fn)).squeeze() for subdir in subdirs]


def probes(subdirs, out_dir, sampling_rate=30000, labels=None):
    """
    Merge spike-sorted data from different probes and output datasets to disk
    :param subdirs: Path or string of full path to the spike-sorted data, each probe in a folder
    :param out_dir: Output directory for merged spike-sorted data
    :param sampling_rate: Hz sampling rate of the AP band binary data
    :param labels: labels for each probe. If not specified, defaults to the containing folder name
     for each probe
    :return: None
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not labels:
        labels = [Path(subdir).parts[-1] for subdir in subdirs]

    # %% Probes, this will have to be spun off to another function to be executed
    # even for single probe data
    with open(out_dir / 'probes.description.txt', 'w+') as fid:
        fid.writelines([f'{lab}\n' for lab in labels])

    # %% Spike times
    spike_times_l = _load_multiple_files('spike_times.npy', subdirs)
    spike_times, order = _load_multiple_spike_times(*spike_times_l)
    np.save(out_dir / 'spike_times.npy', spike_times)

    spike_data = [
        'amplitudes.npy',
        'spike_templates.npy',
        # 'pc_features.npy',
        'template_features.npy',
    ]

    for fn in spike_data:
        arrays = _load_multiple_files(fn, subdirs)
        concat = _load_multiple_spike_arrays(*arrays, spike_order=order)
        print("Saving", fn, concat.shape)
        np.save(out_dir / fn, concat)

    # %% Spike clusters
    spike_clusters_l = _load_multiple_files('spike_clusters.npy', subdirs)
    cluster_offsets = []
    cluster_probes = []
    offset = 0
    ind = 0
    for subdir, sc in zip(subdirs, spike_clusters_l):
        sc += np.uint32(offset)
        cluster_offsets.append(offset)
        cluster_probes.append(np.int8(sc * 0 + ind))
        offset = sc.max() + 1
        ind += 1
    cluster_probes = np.concatenate(cluster_probes, axis=0)
    spike_clusters = _load_multiple_spike_arrays(*spike_clusters_l, spike_order=order)
    np.save(out_dir / 'clusters.probe.npy', cluster_probes)
    np.save(out_dir / 'spike_clusters.npy', spike_clusters)
    np.save(out_dir / 'spike_templates.npy', spike_clusters)

    # ref = np.load(subdirs[0] / 'spike_clusters.npy')
    # test = spike_clusters[order]
    # tref = np.load(subdirs[0] / 'spike_times.npy')
    #
    # inds = spike_clusters < cluster_offsets[1]
    # np.all(spike_clusters[inds] == ref.squeeze())
    # np.all(spike_times[inds] == tref.squeeze())
    #
    # np.all(spike_clusters_l[0] == ref.squeeze())
    # np.all(spike_clusters_l[1] >= cluster_offsets[1])
    # np.all(spike_clusters_l[0] < cluster_offsets[1])

    # %% Cluster-dependent data
    """ We load all cluster metadata from TSV files, renumber the clusters, merge the dictionaries,
     and save in a new merged TSV file. """

    cluster_data = [
        'cluster_Amplitude.tsv',
        'cluster_ContamPct.tsv',
        'cluster_KSLabel.tsv'
    ]

    for fn in cluster_data:
        metadata = {}
        for subdir, offset in zip(subdirs, cluster_offsets):
            field_name, metadata_loc = _read_tsv_simple(subdir / fn)
            for k, v in metadata_loc.items():
                metadata[k + offset] = v
        _write_tsv_simple(out_dir / fn, field_name, metadata)

    # %% Channel-dependent data

    # channel maps
    channel_offsets = []
    channel_probes = []
    channel_maps_l = _load_multiple_files('channel_map.npy', subdirs)
    offset = 0
    for ind, array in enumerate(channel_maps_l):
        array += offset
        channel_offsets.append(offset)
        offset = array.max()
        channel_probes.append(array * 0 + ind)
    channel_maps = np.concatenate(channel_maps_l, axis=0)
    channel_probes = np.concatenate(channel_probes, axis=0)
    print("Saving channel_maps and probes", channel_maps.shape)
    np.save(out_dir / 'channel_map.npy', channel_maps)
    np.save(out_dir / 'channel_probe.npy', channel_probes)

    # channel positions
    channel_positions_l = _load_multiple_files('channel_positions.npy', subdirs)
    x_offset = 0.
    for array in channel_positions_l:
        array[:, 0] += x_offset
        x_offset = 2. * array[:, 0].max() - array[:, 0].min()
    channel_positions = np.concatenate(channel_positions_l, axis=0)
    print("Saving channel_positions", channel_positions.shape)
    np.save(out_dir / 'channel_positions.npy', channel_positions)

    # %% Template-dependent data
    """Templates.npy: the output of KS2 is dense. We need to convert to a sparse array first.
    TODO..."""

    templates_l = _load_multiple_files('templates.npy', subdirs)

    n_templates = sum(tmp.shape[0] for tmp in templates_l)
    n_samples = templates_l[0].shape[1]  # assuming all have the same number of samples
    n_channels = sum(tmp.shape[2] for tmp in templates_l)

    templates = np.zeros((n_templates, n_samples, n_channels))

    for i in range(len(subdirs)):
        i0 = templates_l[i - 1].shape[0] if i > 0 else 0
        i1 = i0 + templates_l[i].shape[0]
        j0 = templates_l[i - 1].shape[2] if i > 0 else 0
        j1 = j0 + templates_l[i].shape[2]
        templates[i0:i1, :, j0:j1] = templates_l[i]
    np.save(out_dir / 'templates.npy', templates)

    template_data = [
        # 'templates_ind.npy',  # HACK: do not copy this array (which is trivial with 0 1 2 3...
        # on each row),
        # the templates.npy file is really dense in KS2 and should stay this way
        'pc_feature_ind.npy',
        'template_feature_ind.npy',
    ]

    for fn in template_data:
        arrays = _load_multiple_files(fn, subdirs)
        # For ind arrays, we need to take into account the channel offset.
        for array, offset in zip(arrays, channel_offsets):
            array += offset
        concat = np.concatenate(arrays, axis=0)
        print("Saving", fn, concat.shape)
        # HACK: templates_ind is not in uint32 for some reason, enforcing it here
        np.save(out_dir / fn, np.asarray(concat, dtype=np.uint32))

    # %% Special data
    """
    Similar templates: we make a block diagonal matrix from then_templates * n_templates matrices,
     assuming no similarity between templates from different probes.

    Whitening matrix: same thing, except that the matrices are n_channels * n_channels.

    """
    diag_data = [
        'similar_templates.npy',
        'whitening_mat.npy',
        'whitening_mat_inv.npy',
    ]
    for fn in diag_data:
        concat = block_diag(*_load_multiple_files(fn, subdirs))
        print("Saving", fn, concat.shape)
        np.save(out_dir / fn, concat)

    # output a template model - also serves the purpose of minimal consistency check...
    m = model.TemplateModel(dir_path=out_dir,
                            dat_path=None,
                            sample_rate=sampling_rate,
                            n_channels_dat=np.max(channel_maps) + 1)
    return m
