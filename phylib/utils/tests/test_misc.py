# -*- coding: utf-8 -*-

"""Tests of misc utility functions."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from numpy.testing import assert_array_equal as ae
from pytest import raises, mark

from .._misc import (
    _git_version, load_json, save_json, load_pickle, save_pickle, read_python, write_python,
    read_text, write_text, _read_tsv_simple, _write_tsv_simple, read_tsv, write_tsv,
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


def test_git_version():
    v = _git_version()
    assert v


def _myfunction(x):
    return


def test_fullname():
    assert _fullname(_myfunction) == 'phylib.utils.tests.test_misc._myfunction'

    assert _load_from_fullname(_myfunction) == _myfunction
    assert _load_from_fullname(_fullname(_myfunction)) == _myfunction
