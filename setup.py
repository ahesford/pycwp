#!/usr/bin/env python
"""
pycwp: Userful numerical routines for computational wave physics in Pyhon
The pycwp library is maintained by Andrew J. Hesford to provide useful
software in Python for computational wave physics and the manipulation
of binary matrix files.
"""

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

DOCLINES = __doc__.split("\n")
VERSION = '3.1'

if __name__ == '__main__':
	from setuptools import setup, find_packages, Extension
	from Cython.Build import cythonize
	from glob import glob

	import numpy as np

	ext_includes = [ np.get_include() ]
	extensions = [ Extension('*', ['**/*.pyx'], include_dirs=ext_includes) ]

	setup(name="pycwp",
			version=VERSION,
			description=DOCLINES[0],
			long_description="\n".join(DOCLINES[2:]),
			author="Andrew J. Hesford",
			author_email="ajh@sideband.org",
			platforms=["any"],
			classifiers=[
				'License :: OSI Approved :: BSD License',
				'Programming Language :: Python :: 3',
				'Intended Audience :: Developers',
				'Topic :: Scientific/Engineering',
				'Development Status :: 4 - Beta'
			],
			packages=find_packages(),
			scripts=glob('shell/*.py'),
			ext_modules=cythonize(extensions,
				compiler_directives={'embedsignature': True}),
			include_dirs=ext_includes,
			package_data={ 'pycwp.cytools': ['*.pxd'] }
			)
