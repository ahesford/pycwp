# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

# Some portions of this file, as noted, are taken from the pandas module
# available at https://github.com/pandas-dev/pandas/. Those portions are
# subject to the following license:
#
# Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc.
# and PyData Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from this
#     software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import cython
cimport cython

import numpy as np
cimport numpy as np

from libc.math cimport signbit

cdef double NaN = <double>np.NaN

cdef inline void add_mean(double val, long *nobs, double *sum_x, long *neg_ct) nogil:
	'''
	A helper to add a value from rolling_mean. Taken from pandas.
	'''
	if val == val:
		nobs[0] += 1
		sum_x[0] += val
		if signbit(val): neg_ct[0] += 1

cdef inline void rem_mean(double val, long *nobs, double *sum_x, long *neg_ct) nogil:
	'''
	A helper to remove a value from rolling_mean. Taken from pandas.
	'''
	if val == val:
		nobs[0] -= 1
		sum_x[0] -= val
		if signbit(val): neg_ct[0] -= 1

@cython.cdivision(True)
cdef inline double calc_mean(long nobs, double sum_x, long neg_ct) nogil:
	'''
	A helper to evaluate the running mean from rolling_mean.

	Adapted from pandas.
	'''
	cdef double result
	result = sum_x / <double>nobs
	# Make sure all negative or all positive values produce sensible means
	if neg_ct == 0 and result < 0: result = 0
	elif neg_ct == nobs and result > 0: result = 0
	return result

@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_mean(a, unsigned long n):
	'''
	Compute the rolling average of width n of a, which must be compatible
	with a 1-D floating-point Numpy array.

	Values of n must fall in the interval [0, a.shape[0]] and, as a special
	case, the value 0 has the same interpretation as the value a.shape[0].

	The return value is a 1-D double-precision array out of length len(a)
	such that

	  out[i] = mean(x[:i+1]) for 0 <= i < n, (the "expanding" region)
	  out[i] = mean(x[i-n:i]) for n <= i < len(x) (the "rolling" region).
	'''
	cdef:
		np.ndarray[np.float64_t, ndim=1] arr
		np.ndarray[np.float64_t, ndim=1] out
		double sum_x = 0
		long i, nobs = 0, neg_ct = 0

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty_like(arr)

	with nogil:
		for i in range(n):
			# Accumulate the values in the growing window region
			add_mean(arr[i], &nobs, &sum_x, &neg_ct)
			out[i] = calc_mean(nobs, sum_x, neg_ct)

		for i in range(n, arr.shape[0]):
			# Accumulate and remove values in the sliding window region
			add_mean(arr[i], &nobs, &sum_x, &neg_ct)
			rem_mean(arr[i - n], &nobs, &sum_x, &neg_ct)
			out[i] = calc_mean(nobs, sum_x, neg_ct)

	return out

@cython.cdivision(True)
cdef inline void add_var(double val, long *nobs,
			double *mean_x, double *ssqdm_x) nogil:
	'''
	A helper to add a value from rolling_var. Taken from pandas.
	'''
	cdef double delta, dobs

	if val == val:
		nobs[0] += 1
		dobs = <double>(nobs[0])

		delta = val - mean_x[0]
		mean_x[0] += delta / dobs
		ssqdm_x[0] += ((dobs - 1) * delta**2) / dobs

@cython.cdivision(True)
cdef inline void rem_var(double val, long *nobs,
			double *mean_x, double *ssqdm_x) nogil:
	'''
	A helper to remove a value from rolling_var. Taken from pandas.
	'''
	cdef double delta, dobs

	if val == val:
		nobs[0] -= 1
		dobs = <double>(nobs[0])
		if nobs[0]:
			delta = val - mean_x[0]
			mean_x[0] -= delta / dobs
			ssqdm_x[0] -= ((dobs + 1) * delta**2) / dobs
		else:
			mean_x[0] = 0
			ssqdm_x[0] = 0

@cython.cdivision(True)
cdef inline double calc_var(long ddof, long nobs, double ssqdm_x) nogil:
	'''
	A helper to calculate the variance from rolling_var.

	Adapted from pandas.
	'''
	cdef double result

	if nobs <= ddof: return NaN

	if nobs == 1:
		result = 0
	else:
		result = ssqdm_x / <double>(nobs - ddof)
		if result < 0: result = 0

	return result

@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_var(a, unsigned long n, unsigned long ddof=0):
	'''
	Compute the rolling variance of width n of a. The arguments a and n,
	together with output (r) are interpreted as in rolling_mean, except the
	variance (with ddof degrees of freedom) is computed in place of the
	mean. Any output indices i <= ddof will have a NaN value.

	Adapted from pandas.
	'''
	cdef:
		np.ndarray[np.float64_t, ndim=1] arr
		np.ndarray[np.float64_t, ndim=1] out
		double val, prev, delta, dobs, mean_x_old, mean_x = 0, ssqdm_x = 0
		long i, nobs = 0

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty_like(arr)

	with nogil:
		for i in range(n):
			# Accumulate the values in the growing window region
			add_var(arr[i], &nobs, &mean_x, &ssqdm_x)
			out[i] = calc_var(ddof, nobs, ssqdm_x)

		for i in range(n, arr.shape[0]):
			val = arr[i]
			prev = arr[i - n]

			# Handle simultaneous add and remove
			if val == val:
				if prev == prev:
					delta = val - prev
					mean_x_old = mean_x
					dobs = <double>nobs

					mean_x += delta / dobs
					ssqdm_x += delta * ((dobs - 1) * val
							+ (dobs + 1) * prev
							- 2 * nobs * mean_x_old) / dobs
				else:
					add_var(arr[i], &nobs, &mean_x, &ssqdm_x)
			elif prev == prev:
				rem_var(arr[i - n], &nobs, &mean_x, &ssqdm_x)

			out[i] = calc_var(ddof, nobs, ssqdm_x)

	return out
