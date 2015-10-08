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

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None,  parent_package, top_path)
	config.set_options(ignore_setup_xxx_py=True,
			assume_default_configuration=True,
			delegate_options_to_subpackages=True,
			quiet=True)
	config.add_subpackage('pycwp')
	config.add_scripts(['shell/*.py'])

	return config

if __name__ == '__main__': 
	from numpy.distutils.core import setup, Extension

	setup(name = "pycwp", version = "2.15",
		description = DOCLINES[0],
		long_description = "\n".join(DOCLINES[2:]),
		author = "Andrew J. Hesford", author_email = "ahesford@mac.com",
		platforms = ["any"], license = "BSD", packages = ["pycwp"],
		configuration=configuration)
