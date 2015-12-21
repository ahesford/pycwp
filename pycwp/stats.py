'''
Routines used for basic statistical analysis.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy
from numpy import ma
from itertools import izip


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
		k, s = zip(*s.items())
	except AttributeError:
		s = numpy.asarray(s)

	# Calculate the quartiles and IQR
	q1, q2, q3 = numpy.percentile(s, [25, 50, 75])
	iqr = q3 - q1
	lo, hi = q1 - m * iqr, q3 + m * iqr

	try:
		return dict(kp for kp in izip(k, s) if lo <= kp[1] <= hi)
	except NameError:
		return ma.MaskedArray(s, numpy.logical_or(s < lo, s > hi))


def binomial(n, k):
	'''
	Compute the binomial coefficient n choose k.
	'''
	return prod(float(n - (k - i)) / i for i in range(1, k+1))


def rolling_mean(x, n):
	'''
	Compute the rolling mean of length n for a sequence x. If the sequence
	is not 1-D, it will be flattened first.
	'''
	# Compute the cumulative sum
	ret = numpy.cumsum(x)
	# Subtract too-early contributions
	ret[n:] = ret[n:] - ret[:-n]

	if not numpy.issubdtype(ret.dtype, numpy.inexact):
		ret = ret.astype(numpy.float64)

	return ret[n - 1:] / n


def rolling_variance(x, n):
	'''
	Compute the variance, over a sliding window of length n, for the
	sequence x. If the sequence is not 1-D, it will be flattened first.
	'''
	x = numpy.asarray(x)
	return rolling_mean(x**2, n) - rolling_mean(x, n)**2
