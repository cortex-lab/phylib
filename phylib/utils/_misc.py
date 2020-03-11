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
import os
from pathlib import Path
import subprocess
from textwrap import dedent

import numpy as np

from ._types import _is_integer

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# JSON utility functions
#------------------------------------------------------------------------------

def _encode_qbytearray(arr):
    """Encode binary arrays with base64."""
    b = arr.toBase64().data()
    data_b64 = base64.b64encode(b).decode('utf8')
    return data_b64


def _decode_qbytearray(data_b64):
    """Decode binary arrays with base64."""
    encoded = base64.b64decode(data_b64)
    try:
        from PyQt5.QtCore import QByteArray
        out = QByteArray.fromBase64(encoded)
    except ImportError:  # pragma: no cover
        pass
    return out


class _CustomEncoder(json.JSONEncoder):
    """JSON encoder that accepts NumPy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim == 1 and obj.shape[0] <= 10:
            # Serialize small arrays in clear text (lists of numbers).
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            obj_contiguous = np.ascontiguousarray(obj)
            data_b64 = base64.b64encode(obj_contiguous.data).decode('utf8')
            return dict(__ndarray__=data_b64, dtype=str(obj.dtype), shape=obj.shape)
        elif obj.__class__.__name__ == 'QByteArray':
            return {'__qbytearray__': _encode_qbytearray(obj)}
        elif isinstance(obj, np.generic):
            return obj.item()
        return super(_CustomEncoder, self).default(obj)  # pragma: no cover


def _json_custom_hook(d):
    """Serialize NumPy arrays."""
    if isinstance(d, dict) and '__ndarray__' in d:
        data = base64.b64decode(d['__ndarray__'])
        return np.frombuffer(data, d['dtype']).reshape(d['shape'])
    elif isinstance(d, dict) and '__qbytearray__' in d:
        return _decode_qbytearray(d['__qbytearray__'])
    return d


def _intify_keys(d):
    """Make sure all integer strings in a dictionary are converted into integers."""
    assert isinstance(d, dict)
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        out[k] = v
    return out


def _stringify_keys(d):
    """Make sure all integers in a dictionary are converted into strings."""
    assert isinstance(d, dict)
    out = {}
    for k, v in d.items():
        if _is_integer(k):
            k = str(k)
        out[k] = v
    return out


def _pretty_floats(obj, n=2):
    """Display floating point numbers properly."""
    if isinstance(obj, (float, np.float64, np.float32)):
        return ('%.' + str(n) + 'f') % obj
    elif isinstance(obj, dict):
        return dict((k, _pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return list(map(_pretty_floats, obj))
    return obj


def load_json(path):
    """Load a JSON file."""
    path = Path(path)
    if not path.exists():
        raise IOError("The JSON file `{}` doesn't exist.".format(path))
    contents = path.read_text()
    if not contents:
        return {}
    out = json.loads(contents, object_hook=_json_custom_hook)
    return _intify_keys(out)


def save_json(path, data):
    """Save a dictionary to a JSON file.

    Support NumPy arrays and QByteArray objects. NumPy arrays are saved as base64-encoded strings,
    except for 1D arrays with less than 10 elements, which are saved as a list for human
    readability.

    """
    assert isinstance(data, dict)
    data = _stringify_keys(data)
    path = Path(path)
    ensure_dir_exists(path.parent)
    with path.open('w') as f:
        json.dump(data, f, cls=_CustomEncoder, indent=2, sort_keys=True)


#------------------------------------------------------------------------------
# Other read/write functions
#------------------------------------------------------------------------------

def load_pickle(path):
    """Load a pickle file using joblib."""
    from joblib import load
    return load(path)


def save_pickle(path, data):
    """Save data to a pickle file using joblib."""
    from joblib import dump
    return dump(data, path)


def read_python(path):
    """Read a Python file.

    Parameters
    ----------

    path : str or Path

    Returns
    -------

    metadata : dict
        A dictionary containing all variables defined in the Python file (with `exec()`).

    """
    path = Path(path)
    if not path.exists():  # pragma: no cover
        raise IOError("Path %s does not exist.", path)
    contents = path.read_text()
    metadata = {}
    exec(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata


def write_python(path, data):
    """Write a dictionary in a Python file.

    Parameters
    ----------

    path : str or Path
        Path to the Python file to write.
    data : dict
        A key-value mapping to write as a Python file.

    Returns
    -------

    """
    with open(path, 'w') as f:
        for k, v in data.items():
            if isinstance(v, str):
                v = '"%s"' % v
            f.write('%s = %s\n' % (k, str(v)))


def read_text(path):
    """Read a text file."""
    path = Path(path)
    return path.read_text()


def write_text(path, contents):
    """Write a text file."""
    contents = dedent(contents)
    path = Path(path)
    ensure_dir_exists(path.parent)
    path.write_text(contents)


def _try_make_number(value):
    """Convert a string into an int or float if possible, otherwise do nothing."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
    raise ValueError()


def read_tsv(path):
    """Read a CSV/TSV file.

    Returns
    -------

    data : list of dicts

    """
    path = Path(path)
    data = []
    if not path.exists():
        logger.debug("%s does not exist, skipping.", path)
        return data
    # Find whether the delimiter is tab or comma.
    with path.open('r') as f:
        delimiter = '\t' if '\t' in f.readline() else ','
    with path.open('r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header.
        field_names = list(next(reader))
        for row in reader:
            data.append({k: _try_make_number(v) for k, v in zip(field_names, row) if v != ''})
    logger.log(5, "Read %s.", path)
    return data


def write_tsv(path, data, first_field=None, exclude_fields=(), n_significant_figures=4):
    """Write a CSV/TSV file.

    Parameters
    ----------

    data : list of dicts
    first_field : str
        The name of the field that should come first in the file.
    exclude_fields : list-like
        Fields present in the data that should not be saved in the file.
    n_significant_figures : int
        Number of significant figures used for floating-point numbers in the file.

    """
    path = Path(path)
    ensure_dir_exists(path.parent)
    delimiter = '\t' if path.suffix == '.tsv' else ','
    with path.open('w', newline='') as f:
        if not data:
            logger.info("Data was empty when writing %s.", path)
            return
        # Get the union of all keys from all rows.
        fields = set().union(*data)
        # Remove ignored fields.
        for field in exclude_fields:
            if field in fields:
                fields.remove(field)
        # Make sure the first field is the first one.
        if first_field in fields:
            fields.remove(first_field)
            fields = [first_field] + sorted(fields)
        else:
            fields = sorted(fields)
        writer = csv.writer(f, delimiter=delimiter)
        # Write the header.
        writer.writerow(fields)
        # Write all rows.
        writer.writerows(
            [[_pretty_floats(row.get(field, None), n_significant_figures)
             for field in fields] for row in data])
    logger.debug("Wrote %s.", path)


def _read_tsv_simple(path):
    """Read a CSV/TSV file with only two columns: cluster_id and <field>.

    Return (field_name, dictionary {cluster_id: value}).

    """
    path = Path(path)
    data = {}
    if not path.exists():
        logger.debug("%s does not exist, skipping.", path)
        return data
    # Find whether the delimiter is tab or comma.
    with path.open('r') as f:
        delimiter = '\t' if '\t' in f.readline() else ','
    with path.open('r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        # Skip the header.
        _, field_name = next(reader)
        for row in reader:
            cluster_id, value = row
            cluster_id = int(cluster_id)
            data[cluster_id] = _try_make_number(value)
    logger.debug("Read %s.", path)
    return field_name, data


def _write_tsv_simple(path, field_name, data):
    """Write a CSV/TSV file with two columns: cluster_id and <field>.

    data is a dictionary {cluster_id: value}.

    """
    path = Path(path)
    ensure_dir_exists(path.parent)
    delimiter = '\t' if path.suffix == '.tsv' else ','
    with path.open('w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(['cluster_id', field_name])
        writer.writerows([(cluster_id, data[cluster_id]) for cluster_id in sorted(data)])
    logger.debug("Wrote %s.", path)


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

def _fullname(o):
    """Return the fully-qualified name of a function."""
    return o.__module__ + "." + o.__name__ if o.__module__ else o.__name__


def _load_from_fullname(name):
    """Load a Python object from its fully qualified name."""
    if not isinstance(name, str):
        return name
    parts = name.rsplit('.', 1)
    return getattr(import_module(parts[0]), parts[1], parts[1])


def _git_version():
    """Return the git version."""
    curdir = os.getcwd()
    os.chdir(str(Path(__file__).parent))
    try:
        with open(os.devnull, 'w') as fnull:
            version = ('-git-' + subprocess.check_output(
                       ['git', 'describe', '--abbrev=8', '--dirty', '--always', '--tags'],
                       stderr=fnull).strip().decode('ascii'))
            return version
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return ""
    finally:
        os.chdir(curdir)


def phy_config_dir():
    """Return the absolute path to the phy user directory. By default, `~/.phy/`."""
    return Path.home() / '.phy'


def ensure_dir_exists(path):
    """Ensure a directory exists, and create it otherwise."""
    path = Path(path)
    if path.exists():
        assert path.is_dir()
    else:
        path.mkdir(exist_ok=True, parents=True)
    assert path.exists() and path.is_dir()
