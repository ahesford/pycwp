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


def _largest_type(dtype):
	'''
	If dtype is a complex datatype, return numpy.complex128. Otherwise,
	return numpy.float64.
	'''
	if numpy.issubdtype(dtype, numpy.complexfloating):
		return numpy.dtype('complex128')
	else: return numpy.dtype('float64')


def rolling_mean(x, n):
	'''
	Compute the rolling mean of length n along x as a flattened Numpy
	array.
	'''
	if n < 1: raise ValueError('Value of n must be nonnegative')
	x = numpy.asarray(x)
	dtype = _largest_type(x.dtype)

	mx = numpy.mean(x, dtype=dtype)
	ca = numpy.cumsum(x - mx, dtype=dtype)
	ca[n:] -= ca[:-n]
	return (mx + ca[n - 1:] / n).astype(x.dtype, copy=False)


def rolling_var(x, n):
	'''
	Compute the variance, over a sliding window of length n, along x as a
	flattened Numpy array.
	'''
	if n < 1: raise ValueError('Value of n must be nonnegative')
	x = numpy.asarray(x)
	dtype = _largest_type(x.dtype)

	mx = numpy.mean(x, dtype=dtype)
	ca = numpy.cumsum(x - mx, dtype=dtype)
	cb = numpy.cumsum((x - mx)**2, dtype=dtype)

	ca[n:] -= ca[:-n]
	cb[n:] -= cb[:-n]

	return (cb[n - 1:] - ca[n - 1:]**2 / n) / n


def rolling_std(x, n):
	'''
	Compute the standard deviation, over a sliding window of length n,
	along x as a flattened Numpy array.
	'''
	return numpy.sqrt(rolling_var(x, n))


def rolling_rms(x, n):
	'''
	Compute the RMS value, over a sliding window of length n, along the
	last axis of x.
	'''
	return numpy.sqrt(rolling_mean(x**2, n))
