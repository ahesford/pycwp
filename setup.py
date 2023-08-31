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

def get_cpu_count():
	try:
		import multiprocessing
		return multiprocessing.cpu_count()
	except Exception:
		return 0

from setuptools.command.build_ext import build_ext as _build_ext
class build_ext(_build_ext):
	def finalize_options(self):
		super().finalize_options()
		if self.parallel is None:
			self.parallel = get_cpu_count()


if __name__ == '__main__':
	from setuptools import setup, find_packages, Extension
	from glob import glob

	import numpy as np
	import os

	try:
		from Cython.Build import cythonize
	except ImportError:
		def cythonize(*args, **kwargs):
			from cython.Build import cythonize as real_cythonize
			return real_cythonize(*args, **kwargs)

	cython_opts = {
		'language_level': 2,
		'embedsignature': True,
	}

	cyfile = os.environ.get('PYCWP_CYTHON_OPTS', 'cython_opts.json')
	if os.path.isfile(cyfile):
		import json
		with open(cyfile) as f:
			cython_opts.update(json.load(f))

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
				nthreads=get_cpu_count(),
				compiler_directives=cython_opts),
			include_dirs=ext_includes,
			package_data={ 'pycwp.cytools': ['*.pxd'] },
			cmdclass={ 'build_ext': build_ext },
	)
