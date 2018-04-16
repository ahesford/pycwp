'''
Routines used for basic statistical analysis.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy
from numpy import ma

from .cytools.stats import rolling_mean as rma, rolling_var as rva

def mask_outliers(s, m=1.5):
	'''
	Given a NumPy array s, return a NumPy masked array with outliers
	masked. The lower quartile (q1), median (q2), and upper quartile (q3)
	are calculated for s. Outliers are those that fall outside the range

		[q1 - m * IQR, q3 + m * IQR],

	where IQR = q3 - q1 is the interquartile range.

	If s is, instead, a dictionary-like collection (it has an items()
	method that returns key-value pairs), outlying values (as defined
	above) and corresponding keys will simply be removed from a copy of s.
	'''
	try:
		items = s.items()
	except AttributeError:
		s = numpy.asarray(s)
	else:
		try: k, s = list(zip(*items))
		except ValueError: return { }

	# Calculate the quartiles and IQR
	q1, q2, q3 = numpy.percentile(s, [25, 50, 75])
	iqr = q3 - q1
	lo, hi = q1 - m * iqr, q3 + m * iqr

	try:
		return dict(kp for kp in zip(k, s) if lo <= kp[1] <= hi)
	except NameError:
		return ma.MaskedArray(s, numpy.logical_or(s < lo, s > hi))


def binomial(n, k):
	'''
	Compute the binomial coefficient n choose k.
	'''
	return prod(float(n - (k - i)) / i for i in range(1, k+1))


def rolling_window(x, n):
	'''
	Given an N-dimensional array x (or an array-compatible object) with shape
	(s_1, s_2, ..., s_N), return an (N+1)-dimensional view into x with shape
	(s_1, s_2, ..., s_{N-1}, s_{N} - n + 1, n), wherein the N+1 axis is a
	rolling window fo values along the N axis of x. In other words,

		rolling_window(x, n)[t1,t2,...,tN,:] == x[t1,t2,...,tN:tN+n].

	The function numpy.lib.stride_tricks.as_strided is used to create the
	view. For applications, see rolling_mean() and rolling_variance().
	'''
	x = numpy.asarray(x)
	shape = x.shape[:-1] + (x.shape[-1] - n + 1, n)
	strides = x.strides + (x.strides[-1],)
	return numpy.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _validate_rolling_inputs(x, n):
	'''
	Verify that n is None or a nonnegative integer and x can be represented
	as a Numpy array, then return (xa, nv), where xa is the array
	representation of x; and nv is len(xa) if n is None, n otherwise.
	'''
	n = n or 0
	if n < 0 or n != int(n):
		raise ValueError('Value "n" must be None or a nonnegative integer')

	# Make sure the array is writeable and has a floating-point type
	x = numpy.asarray(x)
	if not numpy.issubdtype(x.dtype, numpy.floating):
		x = x.astype('float64')
	if not x.flags['WRITEABLE']: x = x.copy()

	if n is None: n = len(x)
	return x, n


def rolling_mean(x, n, expand=False):
	'''
	Compute the rolling mean of length n along x as a flattened Numpy
	array.

	When expand=False, only the valid region of the rolling mean is
	returned; for R = rolling_mean(x, n, expand=False),
	
	  R[i] = mean(x[i:n+i]), 0 <= i < len(x) - n + 1.

	When expand=True and R = rolling_mean(x, n, expand=True),

	  R[i] = mean(x[:i+1]) for 0 <= i < n,
	  R[i] = mean(x[i-n:i]) for n <= i < len(x).
	'''
	x, n = _validate_rolling_inputs(x, n)
	mx = rma(x, n)
	if not expand: return mx[n - 1:]
	return mx


def rolling_var(x, n, expand=False):
	'''
	Performs rolling computations as described in rolling_mean, except the
	variance is computed in place of the mean.
	'''
	x, n = _validate_rolling_inputs(x, n)

	# Compute cumulative mean and mean squared
	vx = rva(x, n)
	if not expand: return vx[n - 1:]
	return vx


def rolling_std(*args, **kwargs):
	'''
	Convenience function to compute the rolling standard deviation of x as
	sqrt(rolling_var(*args, **kwargs)).
	'''
	return numpy.sqrt(rolling_var(*args, **kwargs))


def rolling_rms(x, *args, **kwargs):
	'''
	Convenience function to compute the rolling RMS of x as
	sqrt(rolling_mean(x**2, *args, **kwargs)).
	'''
	return numpy.sqrt(rolling_mean(x**2, *args, **kwargs))
