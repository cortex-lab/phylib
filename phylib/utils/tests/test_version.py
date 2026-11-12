# -*- coding: utf-8 -*-

"""Tests of package versioning."""

import re
from importlib.metadata import version

import phylib


def test_version_format():
    assert re.match(r'^\d+\.\d+\.\d+(?:\.dev\d+)?$', phylib.__version__)


def test_installed_version_matches_package_version():
    """The packaging metadata reads the version from `phylib.__version__`."""
    assert version('phylib') == phylib.__version__
