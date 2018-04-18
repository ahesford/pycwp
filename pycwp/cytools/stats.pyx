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

from libc.math cimport signbit, frexp, fabs

cdef double NaN = <double>np.NaN

cdef inline double neumaier(double * const s, const double v, double * const c) nogil:
	'''
	To *s, add the value of v and accumulate a Neumaier compensation in *c.
	The value of *c should be 0 upon first invocation in an accumulation.
	This function will update the compensation in *c, which should be
	passed unmodified for subsequent calls.
	
	The compensated sum, *s + *c, is returned.
	'''
	cdef double t = s[0] + v
	if fabs(s[0]) >= fabs(v): c[0] += (s[0] - t) + v
	else: c[0] += (v - t) + s[0]
	s[0] = t
	return t + c[0]


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
			double * const sum_x, double * const sum_c,
			long * const neg_ct) nogil:
	'''
	A helper to add a value from rolling_mean. Taken from pandas.
	'''
	nobs[0] += 1
	neumaier(sum_x, val, sum_c)
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
		double sum_x = 0, sum_c = 0, cs, val, prev
		long i, nobs = 0, neg_ct = 0, nextidx

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty_like(arr)

	with nogil:
		for i in range(n):
			# Accumulate the values in expanding region
			add_mean(arr[i], &nobs, &sum_x, &sum_c, &neg_ct)
			out[i] = calc_mean(nobs, sum_x + sum_c, neg_ct)

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
				cs = neumaier(&sum_x, val - prev, &sum_c)
				# Track count of negative values
				if signbit(val): neg_ct += 1
				if signbit(prev): neg_ct -= 1

				out[nextidx] = calc_mean(nobs, cs, neg_ct)
				nextidx += 1

			# No need to restart when all inputs processed
			if nextidx >= arr.shape[0]: break

			nobs = neg_ct = 0
			sum_x = sum_c = 0
			for i in range(nextidx - n + 1, nextidx + 1):
				add_mean(arr[i], &nobs, &sum_x, &sum_c, &neg_ct)
			out[nextidx] = calc_mean(nobs, sum_x + sum_c, neg_ct)
			nextidx += 1

	return out

cdef inline void add_var(const double val, long * const nobs,
			double * const mean_x, double * const ssqdm_x,
			double * const mean_c, double * const ssqdm_c) nogil:
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
	neumaier(mean_x, delta / dobs, mean_c)
	neumaier(ssqdm_x, (dobs - 1) * delta**2 / dobs, ssqdm_c)

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
	rstmag are interpreted as in rolling_mean. The output is an array of
	double-precision floating-point values with shape (len(a), 2); the
	first column contains the rolling variance and the second contains the
	rolling mean as would be computed by rolling_mean.

	The variance is computed with ddof degrees of freedom; the variance
	column of the output will be NaN for indices i <= ddof.

	Adapted from pandas.
	'''
	cdef:
		np.ndarray[np.float64_t, ndim=1] arr
		np.ndarray[np.float64_t, ndim=2] out
		double mean_x = 0, ssqdm_x = 0, mean_c = 0, ssqdm_c = 0
		double val, prev, delta, dobs, upd, sigma, mu, ssq
		long iprev, i, nobs = 0, nextidx

	arr = np.asarray(a, dtype=np.float64)

	if n > arr.shape[0]:
		raise ValueError('Value of n must be in range [0, len(a)]')
	elif n == 0: n = arr.shape[0]

	out = np.empty((arr.shape[0], 2), dtype=np.float64)

	if n == 0:
		# n == 0 <==> len(a) == 0, nothing to do
		return out
	elif n == 1:
		# When n is unity, variance is zero and mean is input
		out[:,0] = 0
		out[:,1] = arr
		return out

	with nogil:
		for i in range(n):
			# Accumulate the values in expanding region
			add_var(arr[i], &nobs, &mean_x, &ssqdm_x, &mean_c, &ssqdm_c)
			out[i,0] = calc_var(ddof, nobs, ssqdm_x + ssqdm_c)
			out[i,1] = mean_x + mean_c

		# Switch to the rolling region, allowing for resets
		nextidx = n
		while True:
			# Number of observations is constant in rolling region
			dobs = <double>nobs

			# Initial Neumaier-corrected mean for loop
			mu = mean_x + mean_c

			while nextidx < arr.shape[0]:
				val = arr[nextidx]
				prev = arr[nextidx - n]

				# Reset if values are too disparate
				if check_exp(val, prev, rstmag): break

				# Simultaneous addition and removal
				delta = val - prev
				sigma = val + prev

				# Do sum-of-squares first; it depends on old mean
				upd = delta * ((sigma - 2 * mu) - delta / dobs)
				ssq = neumaier(&ssqdm_x, upd, &ssqdm_c)
				# Sum-of-squares should be positive; reset
				if ssq < 0: break

				# Now update the mean
				mu = neumaier(&mean_x, delta / dobs, &mean_c)

				out[nextidx,0] = calc_var(ddof, nobs, ssq)
				out[nextidx,1] = mu
				nextidx += 1

			# There is no need to restart; all input was processed
			if nextidx >= arr.shape[0]: break

			# Restart the accumulation to populate the next index
			nobs = 0
			ssqdm_x = mean_x = ssqdm_c = mean_c = 0

			for i in range(nextidx - n + 1, nextidx + 1):
				add_var(arr[i], &nobs, &mean_x,
						&ssqdm_x, &mean_c, &ssqdm_c)

			out[nextidx,0] = calc_var(ddof, nobs, ssqdm_x + ssqdm_c)
			out[nextidx,1] = mu
			nextidx += 1

	return out
