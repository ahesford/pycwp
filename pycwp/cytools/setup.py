#!/usr/bin/env python

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	from Cython.Build import cythonize
	import os

	config = Configuration('cytools', parent_package, top_path, package_data={ })

	# Put includes to package_data to install them
	config.package_data.update({ k: [ '*.pxd' ] for k in config.package_dir })

	for ext in cythonize([ os.path.join(d, '*.pyx')
			for d in config.package_dir.itervalues() ]):
		config.ext_modules.append(ext)

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup

	# Grab the configuration
	setup(**configuration(top_path='').todict())
