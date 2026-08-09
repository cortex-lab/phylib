# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import csv
import json
import os
import re
import stat

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises, mark

from .._misc import (
    _atomic_open, _git_version, load_json, save_json, load_pickle, save_pickle, read_python,
    write_python, read_text, write_text, _read_tsv_simple, _write_tsv_simple, read_tsv, write_tsv,
    _pretty_floats, _encode_qbytearray, _decode_qbytearray, _fullname, _load_from_fullname)


#------------------------------------------------------------------------------
# Misc tests
#------------------------------------------------------------------------------

def test_qbytearray(tempdir):
    try:
        from PyQt5.QtCore import QByteArray
    except ImportError:  # pragma: no cover
        return
    arr = QByteArray()
    arr.append('1')
    arr.append('2')
    arr.append('3')

    encoded = _encode_qbytearray(arr)
    assert isinstance(encoded, str)
    decoded = _decode_qbytearray(encoded)
    assert arr == decoded

    # Test JSON serialization of QByteArray.
    d = {'arr': arr}
    path = tempdir / 'test'
    save_json(path, d)
    d_bis = load_json(path)
    assert d == d_bis


def test_pretty_float():
    assert _pretty_floats(0.123456) == '0.12'
    assert _pretty_floats([0.123456]) == ['0.12']
    assert _pretty_floats({'a': 0.123456}) == {'a': '0.12'}


def test_json_simple(tempdir):
    d = {'a': 1, 'b': 'bb', 3: '33', 'mock': {'mock': True}}

    path = tempdir / 'test_dir/test'
    save_json(path, d)
    d_bis = load_json(path)
    assert d == d_bis

    path.write_text('')
    assert load_json(path) == {}
    with raises(IOError):
        load_json('%s_bis' % path)


@mark.parametrize('kind', ['json', 'pickle'])
def test_json_numpy(tempdir, kind):
    arr = np.arange(20).reshape((2, -1)).astype(np.float32)
    d = {'a': arr, 'b': arr.ravel()[:10], 'c': arr[0, 0]}

    path = tempdir / 'test'
    f = save_json if kind == 'json' else save_pickle
    f(path, d)

    f = load_json if kind == 'json' else load_pickle
    d_bis = f(path)
    arr_bis = d_bis['a']

    assert arr_bis.dtype == arr.dtype
    assert arr_bis.shape == arr.shape
    ae(arr_bis, arr)

    ae(d['b'], d_bis['b'])
    ae(d['c'], d_bis['c'])


def test_read_python(tempdir):
    path = tempdir / 'mock.py'
    with open(path, 'w') as f:
        f.write("""a = {'b': 1}""")

    assert read_python(path) == {'a': {'b': 1}}


def test_write_python(tempdir):
    data = {'a': 1, 'b': 'hello', 'c': [1, 2, 3]}
    path = tempdir / 'mock.py'

    write_python(path, data)
    assert read_python(path) == data


def test_write_text(tempdir):
    for path in (tempdir / 'test_1',
                 tempdir / 'test_dir/test_2.txt',
                 ):
        write_text(path, 'hello world')
        assert read_text(path) == 'hello world'


def test_write_tsv_simple(tempdir):
    path = tempdir / 'test.tsv'
    assert _read_tsv_simple(path) == {}

    # The read/write TSV functions conserve the types: int, float, or strings.
    data = {2: 20, 3: 30.5, 5: 'hello'}
    _write_tsv_simple(path, 'myfield', data)

    assert _read_tsv_simple(path) == ('myfield', data)


def test_write_tsv(tempdir):
    path = tempdir / 'test.tsv'
    assert read_tsv(path) == []
    write_tsv(path, [])

    data = [{'a': 1, 'b': 2}, {'a': 10}, {'b': 20, 'c': 30.5}]

    write_tsv(path, data)
    assert read_tsv(path) == data

    write_tsv(path, data, first_field='b', exclude_fields=('c', 'd'))
    assert read_text(path)[0] == 'b'
    del data[2]['c']
    assert read_tsv(path) == data


class _Boom(Exception):
    pass


def test_atomic_open(tempdir):
    path = tempdir / 'test_dir/test.txt'

    with _atomic_open(path) as f:
        f.write('hello')
    assert read_text(path) == 'hello'
    # The temporary file must not be left behind.
    assert sorted(p.name for p in path.parent.iterdir()) == ['test.txt']

    # A new file gets the permissions open() would have given it, without reading or changing
    # the process-global umask. Compare it with an ordinary file created in the same directory.
    control = path.parent / 'control.txt'
    control.write_text('control')
    assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)
    control.unlink()

    # The permissions of an existing file are preserved.
    expected_mode = stat.S_IMODE(path.stat().st_mode)
    if os.name != 'nt':
        path.chmod(0o640)
        expected_mode = 0o640
    with _atomic_open(path) as f:
        f.write('world')
    assert read_text(path) == 'world'
    assert stat.S_IMODE(path.stat().st_mode) == expected_mode

    # A failure half-way through leaves the previous file untouched and no leftovers.
    with raises(_Boom):
        with _atomic_open(path) as f:
            f.write('this should never be visible')
            raise _Boom()
    assert read_text(path) == 'world'
    assert sorted(p.name for p in path.parent.iterdir()) == ['test.txt']

    # Binary writes use the same atomic and cleanup path, ready for numpy saves.
    with _atomic_open(path, mode='wb') as f:
        f.write(b'bytes')
    assert path.read_bytes() == b'bytes'


@mark.parametrize('writer', ['write_tsv', '_write_tsv_simple'])
def test_write_tsv_atomic(tempdir, monkeypatch, writer):
    # Curation data such as cluster_group.tsv must survive a crash during a save.
    path = tempdir / 'cluster_group.tsv'
    if writer == 'write_tsv':
        def _write(value):
            write_tsv(path, [{'cluster_id': 2, 'group': value}], first_field='cluster_id')
    else:
        def _write(value):
            _write_tsv_simple(path, 'group', {2: value})

    _write('good')
    before = path.read_bytes()
    assert before
    files_before = sorted(p.name for p in tempdir.iterdir())

    real_writer = csv.writer

    def _failing_writer(f, **kwargs):
        # Mimic a crash after part of the file has been written.
        real_writer(f, **kwargs).writerow(['cluster_id', 'group'])
        raise _Boom()

    monkeypatch.setattr(csv, 'writer', _failing_writer)
    with raises(_Boom):
        _write('mua')

    assert path.read_bytes() == before
    assert sorted(p.name for p in tempdir.iterdir()) == files_before


def test_save_json_atomic(tempdir, monkeypatch):
    path = tempdir / 'test.json'
    save_json(path, {'a': 1})
    before = path.read_bytes()
    assert before
    files_before = sorted(p.name for p in tempdir.iterdir())

    def _failing_dump(data, f, **kwargs):
        f.write('{"a":')
        raise _Boom()

    monkeypatch.setattr(json, 'dump', _failing_dump)
    with raises(_Boom):
        save_json(path, {'a': 2})

    assert path.read_bytes() == before
    assert sorted(p.name for p in tempdir.iterdir()) == files_before


def test_git_version():
    v = _git_version()
    assert re.match(r'^\+git\.[0-9a-f]{8}(?:\.dirty)?$', v)


def _myfunction(x):
    return


def test_fullname():
    assert _fullname(_myfunction) == 'phylib.utils.tests.test_misc._myfunction'

    assert _load_from_fullname(_myfunction) == _myfunction
    assert _load_from_fullname(_fullname(_myfunction)) == _myfunction
