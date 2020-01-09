# -*- coding: utf-8 -*-
# flake8: noqa

"""Installation script."""


#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import os
import os.path as op
from pathlib import Path
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


readme = (Path(__file__).parent / 'README.md').read_text()


# Find version number from `__init__.py` without executing it.
with (Path(__file__).parent / 'phylib/__init__.py').open('r') as f:
    version = re.search(r"__version__ = '([^']+)'", f.read()).group(1)

with open('requirements.txt') as f:
    require = [x.strip() for x in f.readlines() if not x.startswith('git+')]

setup(
    name='phylib',
    version=version,
    license="BSD",
    description='Ephys data analysis for thousands of channels',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Cyrille Rossant',
    author_email='cyrille.rossant@gmail.com',
    url='https://github.com/cortex-lab/phylib',
    packages=_package_tree('phylib'),
    package_dir={'phylib': 'phylib'},
    package_data={
        'phylib': [
            '*.vert', '*.frag', '*.glsl', '*.npy', '*.gz', '*.txt',
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=require,
)
