# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import base64
import csv
from importlib import import_module
import json
import logging
import os.path as op
import os
import subprocess
import sys
from textwrap import dedent

import numpy as np
from six import string_types, exec_

from ._types import _is_integer

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# JSON utility functions
#------------------------------------------------------------------------------

def _encode_qbytearray(arr):
    b = arr.toBase64().data()
    data_b64 = base64.b64encode(b).decode('utf8')
    return data_b64


def _decode_qbytearray(data_b64):
    encoded = base64.b64decode(data_b64)
    try:
        from PyQt5.QtCore import QByteArray
        out = QByteArray.fromBase64(encoded)
    except ImportError:  # pragma: no cover
        pass
    return out


class _CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            obj_contiguous = np.ascontiguousarray(obj)
            data_b64 = base64.b64encode(obj_contiguous.data).decode('utf8')
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        elif obj.__class__.__name__ == 'QByteArray':
            return {'__qbytearray__': _encode_qbytearray(obj)}
        elif isinstance(obj, np.generic):
            return obj.item()
        return super(_CustomEncoder, self).default(obj)  # pragma: no cover


def _json_custom_hook(d):
    if isinstance(d, dict) and '__ndarray__' in d:
        data = base64.b64decode(d['__ndarray__'])
        return np.frombuffer(data, d['dtype']).reshape(d['shape'])
    elif isinstance(d, dict) and '__qbytearray__' in d:
        return _decode_qbytearray(d['__qbytearray__'])
    return d


def _intify_keys(d):
    assert isinstance(d, dict)
    out = {}
    for k, v in d.items():
        if isinstance(k, string_types) and k.isdigit():
            k = int(k)
        out[k] = v
    return out


def _stringify_keys(d):
    assert isinstance(d, dict)
    out = {}
    for k, v in d.items():
        if _is_integer(k):
            k = str(k)
        out[k] = v
    return out


def _load_json(path):
    path = op.realpath(op.expanduser(path))
    if not op.exists(path):
        raise IOError("The JSON file `{}` doesn't exist.".format(path))
    with open(path, 'r') as f:
        contents = f.read()
    if not contents:
        return {}
    out = json.loads(contents, object_hook=_json_custom_hook)
    return _intify_keys(out)


def _save_json(path, data):
    assert isinstance(data, dict)
    data = _stringify_keys(data)
    path = op.realpath(op.expanduser(path))
    with open(path, 'w') as f:
        json.dump(data, f, cls=_CustomEncoder, indent=2, sort_keys=True)


def _load_pickle(path):
    """Load a pickle file using joblib."""
    from joblib import load
    return load(path)


def _save_pickle(path, data):
    """Save data to a pickle file using joblib."""
    from joblib import dump
    return dump(data, path)


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

def _fullname(o):
    """Return the fully-qualified name of a function."""
    return o.__module__ + "." + o.__name__ if o.__module__ else o.__name__


def _load_from_fullname(name):
    parts = name.rsplit('.', 1)
    return getattr(import_module(parts[0]), parts[1])


def _read_python(path):
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def _read_text(path):
    with open(path, 'r') as f:
        return f.read()


def _write_text(path, contents):
    contents = dedent(contents)
    dir_path = op.dirname(path)
    if not op.exists(dir_path):
        os.mkdir(dir_path)
    assert op.isdir(dir_path)
    assert not op.exists(path)
    with open(path, 'w') as f:
        f.write(contents)


def _read_tsv(filename):
    """Read a CSV/TSV file with only two columns: cluster_id and <field>.

    Return (field_name, dictionary {cluster_id: value}).

    """
    filename = str(filename)
    data = {}
    if not op.exists(filename):
        logger.warning("%s does not exist, skipping.", filename)
        return data
    # Find whether the delimiter is tab or comma.
    with open(filename, 'r') as f:
        delimiter = '\t' if '\t' in f.readline() else ','
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header.
        _, field_name = next(reader)
        for row in reader:
            cluster_id, value = row
            cluster_id = int(cluster_id)
            data[cluster_id] = value
    logger.info("Read %s.", filename)
    return field_name, data


def _write_tsv(filename, field_name, data):
    """Write a CSV/TSV file with two columns: cluster_id and <field>.

    data is a dictionary {cluster_id: value}.

    """
    filename = str(filename)
    if sys.version_info[0] < 3:  # pragma: no cover
        file = open(filename, 'wb')
    else:
        file = open(filename, 'w', newline='')
    delimiter = '\t' if filename.endswith('.tsv') else ','
    with file as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(['cluster_id', field_name])
        writer.writerows([(cluster_id, data[cluster_id])
                          for cluster_id in sorted(data)])
    logger.info("Wrote %s.", filename)


def _git_version():
    curdir = os.getcwd()
    filedir, _ = op.split(__file__)
    os.chdir(filedir)
    try:
        with open(os.devnull, 'w') as fnull:
            version = ('-git-' + subprocess.check_output(
                       ['git', 'describe', '--abbrev=8', '--dirty',
                        '--always', '--tags'],
                       stderr=fnull).strip().decode('ascii'))
            return version
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return ""
    finally:
        os.chdir(curdir)


def phy_config_dir():
    """Return the absolute path to the phy user directory."""
    home = op.realpath(op.expanduser('~'))
    return op.join(home, '.phy')


def _ensure_dir_exists(path):
    """Ensure a directory exists."""
    if not op.exists(path):
        os.makedirs(path)
    assert op.exists(path) and op.isdir(path)
