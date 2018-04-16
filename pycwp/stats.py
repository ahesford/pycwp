'''
Routines used for basic statistical analysis.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy
from numpy import ma



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
	Verify that n is None or a positive integer and x can be represented as
	a Numpy array, then return (xa, nv, dtype), where xa is the array
	representation of x; nv is len(xa) if n is None, n otherwise; and dtype
	is complex128 when xa.dtype is a complex type, float64 otherwise.
	'''
	if n is not None and (n < 1 or n != int(n)):
		raise ValueError('Value "n" must be None a nonnegative integer')

	x = numpy.asarray(x)
	if n is None: n = len(x)

	if numpy.issubdtype(x.dtype, numpy.complexfloating):
		dtype = numpy.dtype('complex128')
	else: dtype = numpy.dtype('float64')
	return x, n, dtype


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
	x, n, dtype = _validate_rolling_inputs(x, n)

	mx = numpy.mean(x, dtype=dtype)
	ca = numpy.cumsum(x - mx, dtype=dtype)
	ca[n:] -= ca[:-n]

	nm1 = n - 1
	ca[nm1:] = mx + ca[nm1:] / n

	if not expand: return ca[nm1:].astype(x.dtype, copy=False)

	# In "expand" mode, use expanding window before validity region
	ca[:nm1] = mx + ca[:nm1] / numpy.arange(1, n)
	return ca.astype(x.dtype, copy=False)


def rolling_var(x, n, expand=False):
	'''
	Performs rolling computations as described in rolling_mean, except the
	variance is computed in place of the mean.
	'''
	x, n, dtype = _validate_rolling_inputs(x, n)

	# Compute cumulative mean and mean squared
	mx = numpy.mean(x, dtype=dtype)
	ca = numpy.cumsum(x - mx, dtype=dtype)
	cb = numpy.cumsum((x - mx)**2, dtype=dtype)

	# Convert cumulative values to rolling values
	ca[n:] -= ca[:-n]
	cb[n:] -= cb[:-n]

	# Convert rolling mean-squared values to variance in valid region
	nm1 = n - 1
	cb[nm1:] = (cb[nm1:] - ca[nm1:]**2 / n) / n

	if not expand: return cb[nm1:].astype(x.dtype, copy=False)

	# In "expand" mode, use expand window before validity region
	nv = numpy.arange(1, n)
	cb[:nm1] = (cb[:nm1] -  ca[:nm1]**2 / nv) / nv
	return cb.astype(x.dtype, copy=False)


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
