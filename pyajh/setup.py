#!/usr/bin/env python

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration

	config = Configuration('pyajh', parent_package, top_path)
	config.add_subpackage('opencl')
	config.add_subpackage('f2py')

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
