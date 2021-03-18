# -*- coding: utf-8 -*-

"""Utility functions."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np


#------------------------------------------------------------------------------
# Various Python utility functions
#------------------------------------------------------------------------------

_ACCEPTED_ARRAY_DTYPES = (
    float, np.float32, np.float64, int, np.int8, np.int16, np.uint8, np.uint16,
    np.int32, np.int64, np.uint32, np.uint64, bool)


class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


def _bunchify(b):
    """Ensure all dict elements are Bunch."""
    assert isinstance(b, dict)
    b = Bunch(b)
    for k in b:
        if isinstance(b[k], dict):
            b[k] = Bunch(b[k])
    return b


def _is_list(obj):
    """Return whether an object is a list."""
    return isinstance(obj, list)


def _as_scalar(obj):
    """Return whether an object is a scalar number (integer or floating point number)."""
    if isinstance(obj, np.generic):
        return obj.item()
    assert isinstance(obj, (int, float))
    return obj


def _as_scalars(arr):
    """Make sure a list only contains scalar numbers."""
    return [_as_scalar(x) for x in arr]


def _is_integer(x):
    """Return whether an object is an integer."""
    return isinstance(x, (int, np.generic))


def _is_float(x):
    """Return whether an object is a floating point number."""
    return isinstance(x, (float, np.float32, np.float64))


def _as_list(obj):
    """Ensure an object is a list."""
    if obj is None:
        return None
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, tuple):
        return list(obj)
    elif not hasattr(obj, '__len__'):
        return [obj]
    else:
        return obj


def _is_array_like(arr):
    """Return whether an object is an array or a list."""
    return isinstance(arr, (list, np.ndarray))


def _as_array(arr, dtype=None):
    """Convert an object to a numerical NumPy array.

    Avoid a copy if possible.

    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out


def _as_tuple(item):
    """Ensure an item is a tuple."""
    if item is None:
        return None
    elif not isinstance(item, tuple):
        return (item,)
    else:
        return item
