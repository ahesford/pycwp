#!/usr/bin/env python
"""
PyAJH: Userful numerical routines for Pyhon

The PyAJH library is maintained by Andrew J. Hesford to provide useful
software in Python for computational wave physics and the manipulation
of binary matrix files.
"""

DOCLINES = __doc__.split("\n")

def configuration(parent_package='', top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration(None,  parent_package, top_path)
	config.set_options(ignore_setup_xxx_py=True,
			assume_default_configuration=True,
			delegate_options_to_subpackages=True,
			quiet=True)
	config.add_subpackage('pyajh')
	config.add_scripts(['shell/*.py'])

	return config

if __name__ == '__main__': 
	from numpy.distutils.core import setup, Extension

	setup(name = "pyajh", version = "2.12",
		description = DOCLINES[0],
		long_description = "\n".join(DOCLINES[2:]),
		author = "Andrew J. Hesford", author_email = "ahesford@me.com",
		platforms = ["any"], license = "BSD", packages = ["pyajh"],
		configuration=configuration)
