# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
import re

from setuptools import setup


#------------------------------------------------------------------------------
# Setup
#------------------------------------------------------------------------------

def _package_tree(pkgroot):
    path = op.dirname(__file__)
    subdirs = [op.relpath(i[0], path).replace(op.sep, '.')
               for i in os.walk(op.join(path, pkgroot))
               if '__init__.py' in i[2]]
    return subdirs


curdir = op.dirname(op.realpath(__file__))
with open(op.join(curdir, 'README.md')) as f:
    readme = f.read()


# Find version number from `__init__.py` without executing it.
filename = op.join(curdir, 'phylib/__init__.py')
with open(filename, 'r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)


setup(
    name='phylib',
    version=version,
    license="BSD",
    description='Ephys data analysis for thousands of channels',
    long_description=readme,
    author='Cyrille Rossant',
    author_email='cyrille.rossant at gmail.com',
    url='https://github.com/cortex-lab/phylib',
    packages=_package_tree('phylib'),
    package_dir={'phylib': 'phylib'},
    package_data={
        'phylib': ['*.vert', '*.frag', '*.glsl', '*.npy', '*.gz', '*.txt',
                '*.html', '*.css', '*.js', '*.prb'],
    },
    include_package_data=True,
    keywords='phy,data analysis,electrophysiology,neuroscience',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Framework :: IPython",
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
)
