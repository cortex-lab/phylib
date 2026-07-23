# -*- coding: utf-8 -*-

"""Tests of package versioning."""

import re
import subprocess
import sys
from pathlib import Path

import phylib


def test_version_format():
    assert re.match(r'^\d+\.\d+\.\d+(?:\.dev\d+)?$', phylib.__version__)


def test_setup_version_matches_package_version():
    root = Path(__file__).parents[3]
    version = subprocess.check_output(
        [sys.executable, 'setup.py', '--version'], cwd=str(root)).decode().strip()
    assert version == phylib.__version__
