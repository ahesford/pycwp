#!/usr/bin/env python
"""
Setup script.
"""

from distutils.core import setup

setup(name = "pyajh",
		version = "1.4",
		description = "Useful numerical routines",
		long_description = ("Routines to manipulate binary matrix files " +
			"and scattering patterns, and simple numerical routines."),
		author = "Andrew J. Hesford",
		author_email = "ahesford@me.com",
		platforms = ["any"],
		
		license = "BSD",

		packages = ["pyajh"],
		)
