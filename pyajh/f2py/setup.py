#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os, os.path

	# This lists prefixes for library dependencies
	prefixes = ['/usr/local']
	# Try to add FFTW_DIR to the prefixes
	try: prefixes.append(os.environ['FFTW_DIR'])
	except KeyError: pass

	# Buidl the include and library directories lists
	include_dirs = [os.path.join(p, 'include') for p in prefixes]
	library_dirs = [os.path.join(p, 'lib') for p in prefixes]

	config = Configuration('f2py', parent_package, top_path)
	config.add_extension('pade', sources=['pade.f'],)
	config.add_extension('splitstep', sources=['splitstep.f90'],
			include_dirs=include_dirs, library_dirs=library_dirs,
			libraries=['fftw3f_threads', 'fftw3f'],)

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
