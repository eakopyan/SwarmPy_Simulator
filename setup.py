#!/usr/bin/env python
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 2):
    raise Exception('Only Python 3.2+ is supported')

setup(name='swarmpy_simulator',
      version='0.1',
      description="Simulator for swarms of nanosatellites",
      author='Evelyne Akopyan',
      author_email='evelyne.akopyan@gmail.com',
      url='https://github.com/eakopyan/SwarmPy_Simulator',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False)

      #packages=find_packages(exclude=['ez_setup', 'examples', 'tests', 'release']),
