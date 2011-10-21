#!/usr/bin/env python
"""
PyAJH: Userful numerical routines for Pyhon

The PyAJH library is maintained by Andrew J. Hesford to provide useful
software in Python for computational wave physics and the manipulation
of binary matrix files.
"""

DOCLINES = __doc__.split("\n")

from setuptools import setup

setup(name = "pyajh", version = "2.1.5",
		description = DOCLINES[0],
		long_description = "\n".join(DOCLINES[2:]),
		author = "Andrew J. Hesford", author_email = "ahesford@me.com",
		platforms = ["any"], license = "BSD", packages = ["pyajh"],
		package_data={'pyajh': ['*.mako']}, zip_safe=False,
		)
