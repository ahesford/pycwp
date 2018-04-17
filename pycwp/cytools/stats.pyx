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

from libc.math cimport signbit, frexp

cdef double NaN = <double>np.NaN

cdef inline bint check_exp(const double lval, const double rval, const int jump) nogil:
	'''
	Return True if jump is less than 1 or if the binary exponent of rval is
	at least jump binary orders of magnitude larger than lval, False
	otherwise.
	'''
	cdef int lexp, rexp

	if jump < 1: return True

	frexp(lval, &lexp)
	frexp(rval, &rexp)

	return rexp >= lexp + jump

cdef inline void add_mean(const double val, long * const nobs,
			double * const sum_x, long * const neg_ct) nogil:
	'''
	A helper to add a value from rolling_mean. Taken from pandas.
	'''
	nobs[0] += 1
	sum_x[0] += val
	if signbit(val): neg_ct[0] += 1

cdef inline double calc_mean(const long nobs,
			const double sum_x, const long neg_ct) nogil:
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
def rolling_mean(a, const unsigned long n, const int rstmag=12):
	'''
	Compute the rolling average of width n of a, which must be compatible
	with a 1-D floating-point Numpy array.

	Values of n must fall in the interval [0, a.shape[0]] and, as a special
	case, the value 0 has the same interpretation as the value a.shape[0].

	The return value is a 1-D double-precision array out of length len(a)
	such that

	  out[i] = mean(x[:i+1]) for 0 <= i < n, (the "expanding" region)
	  out[i] = mean(x[i-n:i]) for n <= i < len(x) (the "rolling" region).

	The rolling calculation involves adding and subtracting values in a as
	elements move into and out of the window, respectively. When elements
	have very different magnitudes, catastrophic cancellation can destroy
	accuracy. To help avoid this, the rolling accumulators will be reset
	whenever, for a given window position, the value to be removed is
	larger than the value to be added by the binary order of magnitude
	rstmag. If rstmag is less than 1, the reset will happen for every
	window position, converting the rolling calculation to a less efficient
	but more accurate brute-force approach.
	'''
	cdef:
		np.ndarray[np.float64_t, ndim=1] arr
		np.ndarray[np.float64_t, ndim=1] out
		double sum_x = 0, val, prev
		long i, nobs = 0, neg_ct = 0, nextidx

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty_like(arr)

	with nogil:
		for i in range(n):
			# Accumulate the values in expanding region (don't verify)
			add_mean(arr[i], &nobs, &sum_x, &neg_ct)
			out[i] = calc_mean(nobs, sum_x, neg_ct)

		# Switch to rolling region, allowing for resets
		nextidx = n
		while True: 
			while nextidx < arr.shape[0]:
				# Accumulate and remove values
				val = arr[nextidx]
				prev = arr[nextidx - n]

				# Reset if removal is too large compared to addition
				if check_exp(val, prev, rstmag): break

				# Simultaneously add and remove values
				sum_x += (val - prev)
				# Track count of negative values
				if signbit(val): neg_ct += 1
				if signbit(prev): neg_ct -= 1

				out[nextidx] = calc_mean(nobs, sum_x, neg_ct)
				nextidx += 1

			# No need to restart when all inputs processed
			if nextidx >= arr.shape[0]: break

			nobs = neg_ct = 0
			sum_x = 0
			for i in range(nextidx - n + 1, nextidx + 1):
				add_mean(arr[i], &nobs, &sum_x, &neg_ct)
			out[nextidx] = calc_mean(nobs, sum_x, neg_ct)
			nextidx += 1

	return out

cdef inline void add_var(const double val, long * const nobs,
			double * const mean_x, double * const ssqdm_x) nogil:
	'''
	A helper to add a value from rolling_var. Returns False if verify is
	True and roundoff or catastrophic cancellation may have occurred, True
	otherwise.

	Taken from pandas.
	'''
	cdef double delta, dobs

	nobs[0] += 1
	dobs = <double>(nobs[0])

	delta = val - mean_x[0]
	mean_x[0] += delta / dobs
	ssqdm_x[0] += (dobs - 1) * delta**2 / dobs

cdef inline double calc_var(const long ddof,
			const long nobs, const double ssqdm_x) nogil:
	'''
	A helper to calculate the variance from rolling_var.

	Adapted from pandas.
	'''
	cdef double result

	if nobs <= ddof: return NaN

	if nobs == 1: return 0

	result = ssqdm_x / <double>(nobs - ddof)
	if result < 0: result = 0

	return result

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def rolling_var(a, const unsigned long n,
		const unsigned long ddof=0, const int rstmag=12):
	'''
	Compute the rolling variance of width n of a. The arguments a, n, and
	rstmag, together with output (r) are interpreted as in rolling_mean,
	except the variance (with ddof degrees of freedom) is computed in place
	of the mean. Any output indices i <= ddof will have a NaN value.

	Adapted from pandas.
	'''
	cdef:
		np.ndarray[np.float64_t, ndim=1] arr
		np.ndarray[np.float64_t, ndim=1] out
		double val, prev, mean_x = 0, ssqdm_x = 0, delta, dobs
		long iprev, i, nobs = 0, nextidx

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty_like(arr)

	if n == 0:
		# n == 0 <==> len(a) == 0, nothing to do
		return out
	elif n == 1:
		# When n is unity, variance is identically 0
		out[:] = 0
		return out

	with nogil:
		for i in range(n):
			# Accumulate the values in expanding region (don't verify)
			add_var(arr[i], &nobs, &mean_x, &ssqdm_x)
			out[i] = calc_var(ddof, nobs, ssqdm_x)

		# Switch to the rolling region, allowing for resets
		nextidx = n
		while True:
			# Number of observations is constant in rolling region
			dobs = <double>nobs
			while nextidx < arr.shape[0]:
				val = arr[nextidx]
				prev = arr[nextidx - n]

				# Reset if values are too disparate
				if check_exp(val, prev, rstmag): break

				# Simultaneous addition and removal
				delta = val - prev

				# Sum-of-squares update depends on old mean
				ssqdm_x += delta * (dobs * (val + prev) - delta
							- 2 * dobs * mean_x) / dobs
				# Now update the mean
				mean_x += delta / dobs

				out[nextidx] = calc_var(ddof, nobs, ssqdm_x)
				nextidx += 1

			# There is no need to restart; all input was processed
			if nextidx >= arr.shape[0]: break

			# Restart the accumulation to populate the next index
			nobs = 0
			ssqdm_x = mean_x = 0

			for i in range(nextidx - n + 1, nextidx + 1):
				add_var(arr[i], &nobs, &mean_x, &ssqdm_x)

			out[nextidx] = calc_var(ddof, nobs, ssqdm_x)
			nextidx += 1

	return out
