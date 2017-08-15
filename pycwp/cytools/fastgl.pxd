# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

cdef extern from "fastgl.c":
	ctypedef struct qpstruct:
		double theta, weight
	int glpair(qpstruct *result, size_t n, size_t k) nogil
