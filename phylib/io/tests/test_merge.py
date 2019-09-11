# -*- coding: utf-8 -*-

"""Test probe merging."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from ..merge import Merger
from phylib.io.tests.conftest import _make_dataset


#------------------------------------------------------------------------------
# Merging tests
#------------------------------------------------------------------------------

def test_probe_merge_1(tempdir):
    out_dir = tempdir / 'merged'
    probe_names = ('probe_left', 'probe_right')
    for name in probe_names:
        (tempdir / name).mkdir(exist_ok=True, parents=True)
        _make_dataset(tempdir / name, param='dense', has_spike_attributes=False)
    subdirs = [tempdir / name for name in probe_names]
    m = Merger(subdirs, out_dir)
    m.merge()
