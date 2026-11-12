# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import ast
import base64
from contextlib import contextmanager
import csv
from importlib import import_module
import json
import logging
import operator
import os
from pathlib import Path
import secrets
import stat
import subprocess
from textwrap import dedent

import numpy as np

from ._types import _is_integer

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Atomic file writing
#------------------------------------------------------------------------------

def _copy_file_mode(src, dst):
    """Give `src` the permissions of `dst` when `dst` exists."""
    try:
        mode = stat.S_IMODE(os.stat(dst).st_mode)
    except FileNotFoundError:
        # The exclusive open used for a new temporary file already applied the process umask.
        return
    os.chmod(src, mode)


@contextmanager
def _atomic_open(path, mode='w', **kwargs):
    """Open a file for writing without ever truncating the file it replaces.

    The contents are written to a temporary file in the same directory as the destination, and
    that file is moved into place with `os.replace()` once the `with` block completes, which is
    atomic as long as both paths are on the same filesystem. A reader therefore sees either the
    old file or the new one, never a half-written one. If the block raises, the temporary file
    is removed and any existing file is left untouched.

    """
    path = Path(path)
    ensure_dir_exists(path.parent)
    if not mode.startswith('w'):
        raise ValueError("_atomic_open() only supports write modes")
    exclusive_mode = 'x' + mode[1:]
    # NOTE: the temporary file is hidden and does not end in a data file extension, so that a
    # file left behind by a hard kill is not picked up by the globs used to discover datasets.
    # Opening it with `x` both prevents a name collision and lets the OS apply the current umask,
    # without temporarily changing that process-global setting.
    while True:
        tmp = path.parent / ('.%s.%s.tmp' % (path.name, secrets.token_hex(16)))
        try:
            f = open(tmp, exclusive_mode, **kwargs)
            break
        except FileExistsError:  # pragma: no cover - cryptographically unlikely
            continue
    try:
        with f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        _copy_file_mode(tmp, path)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:  # pragma: no cover
            pass
        raise


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
    with _atomic_open(path) as f:
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


MAX_ITERATIONS = 1_000_000  # total loop iterations allowed in comprehensions / helpers
MAX_SIZE = 1_000_000  # maximum number of elements of any container created

_SAFE_CALLS = {
    'range': range, 'list': list, 'tuple': tuple, 'set': set, 'dict': dict, 'len': len,
    'int': int, 'float': float, 'str': str, 'bool': bool, 'abs': abs, 'min': min, 'max': max,
    'sum': sum, 'sorted': sorted, 'zip': zip, 'enumerate': enumerate, 'round': round}
# NumPy >= 2 scalar reprs (e.g. `np.int64(3)`), unwrapped to their literal argument (never called)
_NUMPY_SCALAR_LITERALS = {
    'bool': (bool,), 'bool_': (bool,),
    **{name: (int,) for name in (
        'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64')},
    **{name: (int, float) for name in ('float16', 'float32', 'float64')},
}
_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b}


class _SafeEvaluator:
    """Evaluate AST expressions without executing arbitrary code.

    Supported: literals, list/tuple/set/dict displays, references to previously assigned
    variables, `+ - * / // %` and unary `+ - not`, comparisons, `and`/`or`, `a if c else b`,
    indexing and slicing, list/set/dict comprehensions and generator expressions (with nested
    `for` and `if` clauses), and calls to the helpers in `_SAFE_CALLS`.
    NumPy scalar reprs (`np.int64(3)`, `np.True_`, also under `numpy`) are unwrapped to their
    literal argument, of the matching type, without calling NumPy (see `_NUMPY_SCALAR_LITERALS`).
    Total iterations and container sizes are bounded by `MAX_ITERATIONS` and `MAX_SIZE`.
    Anything else raises a `ValueError`.
    """

    def __init__(self):
        self.variables = {}
        self.iterations = 0

    def _check_size(self, value):
        if isinstance(value, (str, bytes, list, tuple, set, frozenset, dict, range)) \
                and len(value) > MAX_SIZE:
            raise ValueError(f"Object larger than the {MAX_SIZE} elements limit.")
        return value

    def _tick(self, n=1):
        self.iterations += n
        if self.iterations > MAX_ITERATIONS:
            raise ValueError(f"More than {MAX_ITERATIONS} iterations.")

    def _iter(self, iterable):
        # Bound iteration over lazy objects (range, zip...) as well as sequences.
        for x in iterable:
            self._tick()
            yield x

    def _bind(self, target, value, env):
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            value = list(value)
            if len(value) != len(target.elts):
                raise ValueError("Cannot unpack: wrong number of values.")
            for t, v in zip(target.elts, value):
                self._bind(t, v, env)
        else:
            raise ValueError("Unsupported assignment target.")

    def _comprehension(self, node, env):
        """Yield the local namespace of each innermost iteration of a comprehension."""
        def rec(i, env):
            if i == len(node.generators):
                yield env
                return
            gen = node.generators[i]
            if gen.is_async:
                raise ValueError("Async comprehensions are not supported.")
            for x in self._iter(self.eval(gen.iter, env)):
                inner = dict(env)
                self._bind(gen.target, x, inner)
                if all(self.eval(cond, inner) for cond in gen.ifs):
                    yield from rec(i + 1, inner)
        return rec(0, env)

    def eval(self, node, env=None):  # noqa: C901
        """Evaluate an expression node, with `env` holding the comprehension-local names."""
        env = self.variables if env is None else env
        ev = lambda n: self.eval(n, env)  # noqa: E731
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in env:  # `env` always includes the top-level variables
                return env[node.id]
            if node.id in _SAFE_CALLS:
                raise ValueError(f"`{node.id}` can only be called.")
            raise ValueError(f"Undefined name `{node.id}`.")
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if any(isinstance(e, ast.Starred) for e in node.elts):
                raise ValueError("Starred expressions are not supported.")
            container = {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)]
            return self._check_size(container(ev(e) for e in node.elts))
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise ValueError("Dict unpacking is not supported.")
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            a, b = ev(node.left), ev(node.right)
            # Check sizes *before* repeating, to avoid allocating huge sequences.
            if isinstance(node.op, ast.Mult):
                for seq, n in ((a, b), (b, a)):
                    if isinstance(seq, (str, bytes, list, tuple)) and isinstance(n, int) \
                            and len(seq) * n > MAX_SIZE:
                        raise ValueError(f"Object larger than the {MAX_SIZE} elements limit.")
            return self._check_size(_BINOPS[type(node.op)](a, b))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.BoolOp):
            for v in node.values:  # short-circuit and return the operand, as in Python
                value = ev(v)
                if bool(value) == isinstance(node.op, ast.Or):
                    break
            return value
        if isinstance(node, ast.Compare) and all(type(o) in _CMPOPS for o in node.ops):
            left = ev(node.left)
            for op, comp in zip(node.ops, node.comparators):
                right = ev(comp)
                if not _CMPOPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return ev(node.body) if ev(node.test) else ev(node.orelse)
        if isinstance(node, ast.Subscript):
            return ev(node.value)[ev(node.slice)]
        if isinstance(node, ast.Slice):
            return slice(*(ev(x) if x is not None else None
                           for x in (node.lower, node.upper, node.step)))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            out = set() if isinstance(node, ast.SetComp) else []
            add = out.add if isinstance(out, set) else out.append
            for e in self._comprehension(node, env):
                add(self.eval(node.elt, e))
                self._check_size(out)
            return out
        if isinstance(node, ast.DictComp):
            out = {}
            for e in self._comprehension(node, env):
                out[self.eval(node.key, e)] = self.eval(node.value, e)
                self._check_size(out)
            return out
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in _SAFE_CALLS and node.func.id not in env \
                and not node.keywords \
                and not any(isinstance(a, ast.Starred) for a in node.args):
            args = [ev(a) for a in node.args]
            if node.func.id == 'range':
                self._tick(len(range(*args)))  # also validates the arguments
            return self._check_size(_SAFE_CALLS[node.func.id](*args))
        func = node.func if isinstance(node, ast.Call) else node
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) \
                and func.value.id in ('np', 'numpy'):
            if func is node and func.attr in ('True_', 'False_'):
                return func.attr == 'True_'
            if func is not node and func.attr in _NUMPY_SCALAR_LITERALS \
                    and len(node.args) == 1 and not node.keywords:
                value = ast.literal_eval(node.args[0])
                if type(value) in _NUMPY_SCALAR_LITERALS[func.attr]:
                    return value
        raise ValueError(f"Unsupported expression `{type(node).__name__}`.")


def read_python(path, *, allow_python_execution=False):
    """Read a Python file (parameters, probe geometry...) and return its variables.

    Parameters
    ----------
    path : str or Path
    allow_python_execution : bool
        If False (default), the file is parsed but never executed. Only top-level
        `name = expression` assignments (and docstrings) are accepted, where expressions may use:
        literals; list/tuple/set/dict displays; previously assigned variables; `+ - * / // %`;
        comparisons, `and`, `or`, `not`, `a if c else b`; indexing and slicing; list/set/dict
        comprehensions (bounded to 1e6 iterations and 1e6 elements); the helpers
        `range list tuple set dict len int float str bool abs min max sum sorted zip enumerate
        round`; and NumPy scalar reprs such as `np.int64(3)` (unwrapped, not executed).
        Anything else raises a `ValueError` with the file name and line number, so that no
        earlier definition is silently kept.
        If True, the file is run with `exec()`: **only use for trusted legacy files**, as it can
        execute arbitrary code. There is no automatic fallback between the two modes.

    Returns
    -------
    metadata : dict
        The variables defined in the file, with lower-cased keys.

    Raises
    ------
    ValueError
        If the file contains an unsupported construct (and `allow_python_execution` is False).
    """
    path = Path(path)
    if not path.exists():  # pragma: no cover
        raise IOError(f"Path {path} does not exist.")
    contents = path.read_text()
    if allow_python_execution:
        logger.warning(f"Executing {path} with allow_python_execution=True: "
                       "this file can run arbitrary Python code.")
        metadata = {}
        exec(compile(contents, str(path), 'exec'), {}, metadata)
        return {k.lower(): v for (k, v) in metadata.items()}
    evaluator = _SafeEvaluator()
    for node in ast.parse(contents, filename=str(path)).body:
        try:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str):
                continue  # docstring
            if not isinstance(node, ast.Assign):
                raise ValueError(f"Unsupported statement `{type(node).__name__}`.")
            value = evaluator.eval(node.value)
            for target in node.targets:
                evaluator._bind(target, value, evaluator.variables)
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError,
                OverflowError, AttributeError) as e:
            raise ValueError(f"{path}:{node.lineno}: {e} Use `allow_python_execution=True` "
                             "for trusted files.") from e
    return {k.lower(): v for (k, v) in evaluator.variables.items()}


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
    with _atomic_open(path) as f:
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
    with _atomic_open(path) as f:
        f.write(contents)


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
    delimiter = '\t' if path.suffix == '.tsv' else ','
    with _atomic_open(path, newline='') as f:
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
    delimiter = '\t' if path.suffix == '.tsv' else ','
    with _atomic_open(path, newline='') as f:
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
    """Return a PEP 440-compatible local version derived from Git."""
    cwd = str(Path(__file__).parent)
    try:
        with open(os.devnull, 'w') as fnull:
            revision = subprocess.check_output(
                ['git', 'rev-parse', '--short=8', 'HEAD'],
                cwd=cwd, stderr=fnull).strip().decode('ascii')
            dirty = subprocess.call(
                ['git', 'diff', '--quiet', 'HEAD', '--'],
                cwd=cwd, stdout=fnull, stderr=fnull)
            return '+git.' + revision + ('.dirty' if dirty else '')
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return ""


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
