'''
General-purpose numerical routines used in other parts of the module.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, math, operator
from itertools import izip, count


def overlap(lwin, rwin):
	'''
	For two windows lwin and rwin, each of the form (start, length), return
	the tuple (lstart, rstart, length), where lstart and rstart are the
	starts of the overlapping window relative to the starts of lwin and
	rwin, respectively, and length is the length of the overlapping window.

	If there is no overlap in lwin and rwin, None is returned.
	'''
	start = max(lwin[0], rwin[0])
	end = min(lwin[0] + lwin[1], rwin[0] + rwin[1])

	if end <= start: return None
	return (start - lwin[0], start - rwin[0], end - start)


def asarray(a, rank=None, tailpad=True):
	'''
	Ensure that a is an array and has the specified rank. If rank is
	greater than the "natural" rank of a, insert additional axes at the
	beginning (if tailpad is False) or the end (if tailpad is True) of the
	nonempty axes. It is an error to specify a rank less than the "natural"
	rank of a.

	If rank is None, the rank of a is not changed.
	'''
	if a is None: return None

	from numpy import asarray as npasarray, newaxis

	a = npasarray(a)
	nrank = a.ndim

	if rank is None or rank == nrank: return a

	if rank < nrank:
		raise ValueError('Rank must not be less than "natural" rank of a')

	sl = [slice(None)] * nrank + [newaxis] * (rank - nrank)
	if not tailpad: sl = list(reversed(sl))
	return a[sl]


def indexchoose(n, c, s=0):
	'''
	A coroutine to produce successive unique (without regard to order)
	c-tuples of values in an array range(s, n).
	'''
	# Ensure n and c are integers and properly specified
	n = int(n)
	if n < 1:
		raise ValueError('n must be a positive integer')
	s = int(s)
	if s < 0 or s > n:
		raise ValueError('s must be a positive integer between 0 and n')
	c = int(c)
	if c < 0 or c > n - s:
		raise ValueError('c must be an integer between 0 and n - s')

	# If c is 0, return an empty list
	if c == 0:
		yield []
		return
	# If c is n - s, return the only possible selection
	if c == n - s:
		yield list(range(s, n))
		return

	# Otherwise, recursively select the values
	for i in range(s, n - c + 1):
		for vi in indexchoose(n, c - 1, i + 1):
			yield [i] + vi


def vecnormalize(x, ord=None, axis=None):
	'''
	Compute the norm, using numpy.linalg.norm, of the array x and scale x
	by the reciprocal of the norm. The arguments x, ord, and axis must be
	acceptable to numpy.linalg.norm.

	If the norm of x is smaller than the epsilon parameter for the datatype
	of x, the reciprocal of the epsilon parameter will be used to scale x.

	The norm will be broadcast to the proper shape for normalization of x.
	'''
	from numpy import asarray as npasarray, finfo, fmax
	from numpy.linalg import norm

	# Grab the dtype of x (ensuring x is array-compatible)
	x = npasarray(x)

	# Find the smallest representable number for the dtype
	try: eps = finfo(x.dtype).eps
	except ValueError: eps = 1.0

	# Find the norm, but don't allow it to fall below epsilon
	n = fmax(norm(x, ord, axis), eps)

	if axis is None:
		# If no axis was specified, n is a scalar;
		# proper broadcasting will be automatic
		return x / n

	# Ensure that axis is either an integer or a two-integer sequence
	# Treat a single axis integer as a 1-element list
	try: axis = [int(axis)]
	except ValueError: axis = [int(ax) for ax in axis]

	# Prepare the list of slice objects for proper broadcasting
	slicers = [slice(None) for i in range(x.ndim)]
	# Create a new axis for each axis reduced by the norm
	for ax in axis: slicers[ax] = None

	return x / n[slicers]


def waterc(t, p=1.01325):
	'''
	Return the sound speed in water at a temperature of t degrees Celsius
	and an ambient pressure p in bar (1e5 Pa). Units are mm/microsec.

	From "Fundamentals of Acoustics", Kinsler and Frey, et al., Eq. (5.22).
	'''
	# Pressure in bar (1e5 Pa)
	t = t / 100.
	f1 = ((135. * t - 482.) * t + 488.) * t + 1402.7
	f2 = (2.4 * t + 2.8) * t + 15.9
	c = f1 + f2 * p / 100.
	return c / 1000.0


def numdigits(m):
	'''
	Return the number of digits required to represent int(m).
	'''
	if m < 0: raise ValueError('Integer m must be nonnegative')
	elif m == 0: return 1

	digits = int(math.log10(m))
	# Watch for boundary cases (e.g., m = 1000)
	if 10**digits <= m: digits += 1
	return digits


def roundn(x, n):
	'''
	Round x up to the next multiple of n.
	'''
	return x + (n - x) % n


def matchdim(*args):
	'''
	For a list of scalars and sequences, ensure that all provided sequences
	have the same length. For each scalar argument, produce a tuple with
	the value repeated as many times as the length of the sequence
	arguments.

	Copies of all input sequences (converted to tuples) and the repeated
	scalar sequences are returned in the order they were provided.

	A ValueError will be raised if some sequences have disparate lengths.
	'''
	# Check the maximum dimensionality
	def safedim(x):
		try: return len(x)
		except TypeError: return 1
	maxdim = max(safedim(a) for a in args)
	output = []
	for a in args:
		try:
			if len(a) != maxdim:
				raise ValueError('Input arguments must be scalar or have same length')
			else: output.append(tuple(a))
		except TypeError: output.append(tuple([a] * maxdim))

	return tuple(output)


def fuzzyimg(img, nbr):
	'''
	Randomize boundaries in the image represented in the matrix img by
	identifying the maximum and minimum values in the neighborhood of nbr
	pixels per dimension (must be odd) centered around each pixel in the
	image. Assign to the center pixel a uniformly distributed random value
	between the maximum and minimum of the neighborhood.
	'''
	from numpy import fmax, fmin
	from numpy.random import rand
	if nbr % 2 != 1: raise ValueError('Neighborhood must have odd dimensions')
	half = (nbr - 1) / 2
	ndim = len(img.shape)
	# Create the maximum and minimum arrays
	nmax = img.copy()
	nmin = img.copy()
	# Find the limits along each axis successively
	for d in range(ndim):
		# By default, grab the entire array
		rsl = [slice(None)] * ndim
		lsl = [slice(None)] * ndim
		for i in range(1, half + 1):
			# Shift left and right for each offset
			lsl[d] = slice(i, None)
			rsl[d] = slice(None, -i)
			# Grab the max and min in the shifted regions
			nmax[lsl] = fmax(nmax[lsl], img[rsl])
			nmax[rsl] = fmax(nmax[rsl], img[lsl])
			nmin[lsl] = fmin(nmin[lsl], img[rsl])
			nmin[rsl] = fmin(nmin[rsl], img[lsl])

	return rand(*img.shape).astype(img.dtype) * (nmax - nmin) + nmin


def commongrid(lgrid, rgrid):
	'''
	For two grids with dimensions

		lgrid = [lx, ly, ...] and
		rgrid = [rx, ry, ...],

	return a common grid

		cgrid = [cx, cy, ...]

	such that each dimension is the minimum of the corresponding dimsnsions
	in lgrid and rgrid. Also return offsets

		loff = [lox, loy, ...] and
		roff = [rox, roy, ...]

	that indicate the starting coordinates of the common grid in the left
	and right grids, respectively.
	'''

	cgrid = [min(lv, rv) for lv, rv in izip(lgrid, rgrid)]
	loff = [max(0, (lv - rv) / 2) for lv, rv in izip(lgrid, rgrid)]
	roff = [max(0, (rv - lv) / 2) for lv, rv in izip(lgrid, rgrid)]
	return cgrid, loff, roff


def smoothkern(w, s, n = 3):
	'''
	Compute an n-dimensional Gaussian kernel with width w (must be odd) and
	standard deviation s, both measured in pixels. When

		w = 2 * int(4 * s) + 1,

	then convolution with the kernel is equivalent to calling

		scipy.ndimage.gaussian_filter

	with sigma = s.
	'''
	from numpy import zeros, sum as npsum
	from scipy.ndimage import gaussian_filter

	if w % 2 != 1: raise ValueError('Kernel width must be odd.')
	lw = (w - 1) / 2
	# Compute the restricted Gaussian kernel
	k = zeros([w]*n)
	sl = [slice(lw, lw+1)]*n
	k[sl] = 1.
	k = gaussian_filter(k, s, mode='constant')
	return k / npsum(k)


def ceilpow2(x):
	'''
	Find the smallest power of 2 not less than the specified integer x.
	'''
	xc, y = x, 1
	while xc > 1:
		xc >>= 1
		y <<= 1
	return y if y >= x else (y << 1)


def rlocate(arr, val):
	'''
	Search the monotonically increasing or decreasing list arr for an index
	j such that arr[j] <= val <= arr[j+1]. If j == -1 or j == len(arr) - 1,
	val is out of range of the list.

	Return the index.
	'''

	n = len(arr)

	jl = -1;
	ju = n;
	ascnd = arr[-1] >= arr[0]

	while ju - jl > 1:
		jm = (ju + jl) >> 1
		if (val >= arr[jm]) == ascnd: jl = jm
		else: ju = jm

	if val == arr[0]: return 0
	if val == arr[-1]: return n - 1

	return jl


def hadamard(x, y):
	'''
	Compute the Hadamard product of iterables x and y.
	'''
	return tuple(xv * yv for xv, yv in izip(x, y))


def prod(a):
	'''
	Compute the product of the elements of an iterable.
	'''
	return reduce(operator.mul, a)


def dot(x, y):
	'''
	Compute the dot product of two iterables.
	'''
	return sum(xv * yv for xv, yv in izip(x, y))


def cross(x, y):
	'''
	Compute the cross product of two 3-element sequences.
	'''
	xx, xy, xz = x
	yx, yy, yz = y
	return xy * yz - xz * yy, xz * yx - xx * yz, xx * yy - xy * yx


def norm(x):
	'''
	Compute the norm of a vector x.
	'''
	return math.sqrt(sum(abs(xv)**2 for xv in x))


def rotate(x, y = 1):
	'''
	Perform a cyclic rotation of the iterable x.
	'''

	# Don't do anything for a zero-length list
	if len(x) == 0: return x

	# Force x to be a list instead of a tuple or array
	if not isinstance(x, list): x = list(x)

	# Wrap the shift length
	y = y % len(x)

	return x[y:] + x[:y]


def almosteq(x, y, eps=sys.float_info.epsilon):
	'''
	Return True iff the difference between x and y is less than or equal to
	M * eps, where M = max(abs(x), abs(y), 1.0f). The value eps should be
	positive, but this is never enforced.
	'''
	return abs(x - y) <= eps * max(abs(x), abs(y), 1.0)


def mse (x, y):
	'''
	Report the mean squared error between the sequences x and y.
	'''
	n = d = 0.
	for xv, yv in izip(x, y):
		n += abs(xv - yv)**2
		d += abs(yv)**2
	return math.sqrt(n / d)
