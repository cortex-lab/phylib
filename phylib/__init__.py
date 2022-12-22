# -*- coding: utf-8 -*-
# flake8: noqa

"""Utilities for large-scale ephys data analysis."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import atexit
from io import StringIO
import logging
import os.path as op
import sys

from .utils._misc import _git_version
from .utils.event import connect, unconnect, emit


#------------------------------------------------------------------------------
# Global variables and functions
#------------------------------------------------------------------------------

__author__ = 'Cyrille Rossant'
__email__ = 'cyrille.rossant at gmail.com'
__version__ = '2.4.3'
__version_git__ = __version__ + _git_version()


# Set a null handler on the root logger
logger = logging.getLogger('phylib')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


_logger_fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    color_codes = {'L': '94', 'D': '90', 'I': '0', 'W': '33', 'E': '31'}

    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = self.color_codes.get(record.levelname, '90')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


def add_default_handler(level='INFO', logger=logger):
    handler = logging.StreamHandler()
    handler.setLevel(level)

    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def _add_log_file(filename):  # pragma: no cover
    """Create a log file with DEBUG level."""
    handler = logging.FileHandler(str(filename))
    handler.setLevel(logging.DEBUG)
    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)
    logging.getLogger('phy').addHandler(handler)


@atexit.register
def on_exit():  # pragma: no cover
    # Close the logging handlers.
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def test():  # pragma: no cover
    """Run the full testing suite of phylib."""
    import pytest
    pytest.main()
