#!/usr/bin/env python
def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration

	config = Configuration('f2py', parent_package, top_path)
	config.add_extension('pade', sources=['pade.f'],)
	config.add_extension('splitstep', sources=['splitstep.f90'],
			include_dirs=['/usr/local/include'], 
			library_dirs=['/usr/local/lib'],
			libraries=['fftw3f_threads', 'fftw3f'],)

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
