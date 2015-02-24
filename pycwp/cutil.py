'''
General-purpose numerical routines used in other parts of the module.
'''

import numpy, math, operator
from numpy import linalg as la, ma, fft
from scipy import special as spec, ndimage
from itertools import izip


def shifter(sig, delays, s=None, axes=None):
	'''
	Shift a multidimensional signal sig by a number of (possibly
	fractional) sample units in each dimension specified as entries in the
	delays sequence. The shift is done using FFTs and the arguments s and
	axes take the same meaning as in numpy.fft.fftn.

	The length of delays must be compatible with the length of axes as
	specified or inferred.
	'''
	# Ensure that sig is a numpy.ndarray
	sig = asarray(sig)
	ndim = len(sig.shape)

	# Set default values for axes and s if necessary
	if axes is None:
		if s is not None: axes = range(ndim - len(s), ndim)
		else: axes = range(ndim)

	if s is None: s = tuple(sig.shape[a] for a in axes)

	# Check that all arguments agree
	if len(s) != len(axes):
		raise ValueError('FFT shape array and axes list must have same dimensionality')
	if len(s) != len(delays):
		raise ValueError('Delay list and axes list must have same dimensionality')

	# Take the forward transform for spectral shifting
	csig = fft.fftn(sig, s, axes)

	# Loop through the axes, shifting each one in turn
	for d, n, a in zip(delays, s, axes):
		# Build the FFT frequency indices
		dk = 2. * math.pi / n
		kidx = numpy.arange(n)
		k = dk * (kidx >= n / 2.).choose(kidx, kidx - n)
		# Build the shifter and the axis slicer for broadcasting
		sh = numpy.exp(-1j * k * d)
		slic = [numpy.newaxis] * ndim
		slic[a] = slice(None)
		# Multiply the shift
		csig *= sh[slic]

	# Perform the inverse transform and cast to the input type
	rsig = fft.ifftn(csig, axes=axes)
	if not numpy.issubdtype(sig.dtype, numpy.complexfloating):
		rsig = rsig.real
	return rsig.astype(sig.dtype)


def mask_outliers(s, m=1.5):
	'''
	Given a NumPy array s, return a NumPy masked array with outliers
	masked. The lower quartile (q1), median (q2), and upper quartile (q3)
	are calculated for s. Outliers are those that fall outside the range

		[q1 - m * IQR, q3 + m * IQR],

	where IQR = q3 - q1 is the interquartile range.
	'''
	# Calculate the quartiles and IQR
	q1, q2, q3 = numpy.percentile(s, [25, 50, 75])
	iqr = q3 - q1
	lo, hi = q1 - m * iqr, q3 + m * iqr
	return ma.MaskedArray(s, numpy.logical_or(s < lo, s > hi))


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
	# Grab the dtype of x (ensuring x is array-compatible
	try: dtype = x.dtype
	except AttributeError:
		x = numpy.array(x)
		dtype = x.dtype

	# Find the smallest representable number for the dtype
	try: eps = numpy.finfo(dtype).eps
	except ValueError: eps = 1.0

	# Find the norm, but don't allow it to fall below epsilon
	n = numpy.fmax(la.norm(x, ord, axis), eps)

	if axis is None:
		# If no axis was specified, n is a scalar;
		# proper broadcasting will be automatic
		return x / n

	# Ensure that axis is either an integer or a two-integer sequence
	# Treat a single axis integer as a 1-element list
	try: axis = [int(axis)]
	except ValueError: axis = [int(ax) for ax in axis]

	# Prepare the list of slice objects for proper broadcasting
	slicers = [slice(None) for i in range(numpy.ndim(x))]
	# Create a new axis for each axis reduced by the norm
	for ax in axis: slicers[ax] = None

	return x / n[slicers]


def waterc(t):
	'''
	Return the sound speed in water at a temperature of t degrees Celsius.
	Units are mm/microsec.

	From "Fundamentals of Acoustics", Kinsler and Frey, et al., Eq. (5.22).
	'''
	# Pressure in bar (1e5 Pa)
	p = 1.013
	t = t / 100.
	f1 = ((135. * t - 482.) * t + 488.) * t + 1402.7
	f2 = (2.4 * t + 2.8) * t + 15.9
	c = f1 + f2 * p / 100.
	return c / 1000.0


def asarray(a, rank=None, tailpad=True):
	'''
	Ensure that a behaves as a numpy ndarray and, if not, duplicate a as an
	ndarray. If rank is specified, it must not be less than the "natural"
	rank of a. If rank is greater than the natural rank, the returned
	ndarray is padded by appending (if tailpad is True) or prepending (if
	tailpad is False) np.newaxis slices until the desired rank is achieved.

	If a is None, None is returned.

	NOTE: If a is already a numpy array, the returned object is either the
	same as the argument, or a view on the argument.
	'''
	if a is None: return None

	if not isinstance(a, numpy.ndarray): a = numpy.array(a)

	nrank = numpy.ndim(a)
	if rank is None or rank == nrank: return a

	if rank < nrank:
		raise ValueError('Desired rank must not be less than "natural" rank of a')

	sl = [slice(None)] * nrank + [numpy.newaxis] * (rank - nrank)
	if not tailpad: sl = list(reversed(sl))
	return a[sl]


def bandwidth(sigft, df=1, level=0.5, r2c=False):
	'''
	Return as (bw, fc) the bandwidth bw and center frequency fc of a signal
	whose DFT is given in sigft. The frequency bin width is df.

	The DFT is searched in both directions from the positive frequency bin
	with peak amplitude until the signal falls below the specified level.
	Linear interpolation pinpoints the level crossing between bins. The
	bandwidth is the difference between the high and low crossings,
	multiplied by df. Only the level crossing nearest the peak is
	identified in each direction.

	The center frequency is the average of the high and low crossing
	frequencies.

	If r2c is True, the DFT is assumed to contain only positive
	frequencies. Otherwise, the DFT should contain positive and negative
	frequencies in standard FFT order.
	'''
	sigamps = numpy.abs(sigft)
	if not r2c:
		# Strip the negative frequencies from the C2C DFT
		sigamps = sigamps[:len(sigamps)/2]

	# Find the peak positive frequency
	peakidx = numpy.argmax(sigamps)
	# Now scale the amplitudes
	sigamps /= sigamps[peakidx]


	# Search low frequencies for the level crossing
	flo = peakidx + 1
	for i, s in enumerate(reversed(sigamps[:peakidx])):
		if s < level:
			flo = peakidx - i
			break
	# Search high frequencies for the level crossing
	fhi = peakidx - 1
	for i, s in enumerate(sigamps[peakidx+1:]):
		if s < level:
			fhi = peakidx + i
			break

	# Ensure that a crossing level was identified
	if sigamps[flo - 1] > level:
		raise ValueError('Low-frequency level crossing not identified')
	if sigamps[fhi + 1] > level:
		raise ValueError('High-frequency level crossing not identified')

	# Now convert the indices to interpolated frequencies
	# The indices point to the furthest sample exceeding the level
	mlo = (sigamps[flo] - sigamps[flo - 1])
	mhi = (sigamps[fhi + 1] - sigamps[fhi])

	flo = (level - sigamps[flo - 1]) / float(mlo) + flo - 1
	fhi = (level - sigamps[fhi]) / float(mhi) + fhi

	bw = (fhi - flo) * df
	fc = 0.5 * (fhi + flo) * df
	return bw, fc


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


def deg2rad(t):
	'''
	Convert the angle t in degrees to radians.
	'''
	return t * math.pi / 180.


def rad2deg(t):
	'''
	Convert the angle t in radians to degrees.
	'''
	return t * 180. / math.pi


def roundn(x, n):
	'''
	Round x up to the next multiple of n.
	'''
	return x + (n - x) % n


def givens(crd, theta, axes=(0,1)):
	'''
	Perform a Givens rotation of angle theta in the specified axes of the
	coordinate vector crd.
	'''
	c = math.cos(theta)
	s = math.sin(theta)
	ncrd = list(crd)
	x, y = crd[axes[0]], crd[axes[1]]
	ncrd[axes[0]] = x * c - s * y
	ncrd[axes[1]] = x * s + c * y
	return tuple(ncrd)


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
			nmax[lsl] = numpy.fmax(nmax[lsl], img[rsl])
			nmax[rsl] = numpy.fmax(nmax[rsl], img[lsl])
			nmin[lsl] = numpy.fmin(nmin[lsl], img[rsl])
			nmin[rsl] = numpy.fmin(nmin[rsl], img[lsl])

	return numpy.random.rand(*img.shape).astype(img.dtype) * (nmax - nmin) + nmin


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

	cgrid = [min(lv, rv) for lv, rv in zip(lgrid, rgrid)]
	loff = [max(0, (lv - rv) / 2) for lv, rv in zip(lgrid, rgrid)]
	roff = [max(0, (rv - lv) / 2) for lv, rv in zip(lgrid, rgrid)]
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
	if w % 2 != 1: raise ValueError('Kernel width must be odd.')
	lw = (w - 1) / 2
	# Compute the restricted Gaussian kernel
	k = numpy.zeros([w]*n)
	sl = [slice(lw, lw+1)]*n
	k[sl] = 1.
	k = ndimage.gaussian_filter(k, s, mode='constant')
	return k / numpy.sum(k)

def binomial(n, k):
	'''
	Compute the binomial coefficient n choose k.
	'''
	return prod(float(n - (k - i)) / i for i in range(1, k+1))

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
	return map(operator.mul, x, y)

def prod(a):
	'''
	Compute the product of the elements of an iterable.
	'''
	return reduce(operator.mul, a)

def dot(x, y):
	'''
	Compute the dot product of two iterables.
	'''
	return sum(hadamard(x, y))

def lagrange(x, pts):
	'''
	For sample points pts and a test point x, return the Lagrange
	polynomial weights.
	'''

	# Compute a single weight term
	def lgwt(x, pi, pj): return float(x - pj) / float(pi - pj)

	# Compute all the weights
	wts = [prod(lgwt(x, pi, pj) for j, pj in enumerate(pts) if i != j)
			for i, pi in enumerate(pts)]

	return wts

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

def translator (r, s, phi, theta, l):
	'''
	Compute the diagonal translator for a translation distance r, a
	translation direction s, azimuthal samples specified in the array phi,
	polar samples specified in the array theta, and a truncation point l.
	'''

	# The radial argument
	kr = 2. * math.pi * r

	# Compute the radial component
	hl = spec.sph_jn(l, kr)[0] + 1j * spec.sph_yn(l, kr)[0]
	# Multiply the radial component by scale factors in the translator
	m = numpy.arange(l + 1)
	hl *= (1j / 4. / math.pi) * (1j)**m * (2. * m + 1.)

	# Compute Legendre angle argument dot(s,sd) for sample directions sd
	stheta = numpy.sin(theta)[:,numpy.newaxis]
	sds = (s[0] * stheta * numpy.cos(phi)[numpy.newaxis,:]
			+ s[1] * stheta * numpy.sin(phi)[numpy.newaxis,:]
			+ s[2] * numpy.cos(theta)[:,numpy.newaxis])

	# Initialize the translator
	tr = 0

	# Sum the terms of the translator
	for hv, pv in izip(hl, legpoly(sds, l)): tr += hv * pv
	return tr

def legpoly (x, n = 0):
	'''
	A coroutine to generate the Legendre polynomials up to order n,
	evaluated at argument x.
	'''

	if numpy.any(numpy.abs(x) > 1.0):
		raise ValueError("Arguments must be in [-1,1]")
	if n < 0:
		raise ValueError("Order must be nonnegative")

	# Set some recursion values
	cur = x
	prev = numpy.ones_like(x)

	# Yield the zero-order value
	yield prev

	# Yield the first-order value, if that order is desired
	if n < 1: raise StopIteration
	else: yield cur

	# Now compute the subsequent values
	for i in xrange(1, n):
		next = ((2. * i + 1.) * x * cur - i * prev) / (i + 1.)
		prev = cur
		cur = next
		yield cur

	raise StopIteration


def exband (a, tol = 1e-6):
	'''
	Compute the excess bandwidth estimation for an object with radius a
	to a tolerance tol.
	'''

	# Compute the number of desired accurate digits
	d0 = -math.log10(tol)

	# Return the estimated bandwidth
	return int(2. * math.pi * a + 1.8 * (d0**2 * 2. * math.pi * a)**(1./3.))


def legendre (t, m):
	'''
	Return the value of the Legendre polynomial of order m, along with its
	first and second derivatives, at a point t.
	'''
	# Set function values explicitly for orders less than 2
	if m < 1: return 1., 0., 0.
	elif m < 2: return t, 1., 0.

	p0, p1 = 1.0, t

	for k in range(1, m):
		# This value is necessary for the second derivative
		dpl = k * (p0 - t * p1) / (1. - t**2)
		p = ((2.*k + 1.0) * t * p1 - k * p0) / (1. + k)
		p0 = p1; p1 = p

	# Compute the value of the derivative
	dp = m * (p0 - t * p1) / (1.0 - t**2)

	# Compute the value of the second derivative
	ddp = ((m - 2.) * t * dp + m * (p1 - dpl)) / (t**2 - 1.)

	return p, dp, ddp


def clenshaw (m):
	'''
	Compute the Clenshaw-Curtis quadrature nodes and weights in the
	interval [0,pi] for a specified order m.
	'''
	n = m - 1
	idx = numpy.arange(m)
	# Nodes are equally spaced in the interval
	nodes = idx * math.pi / float(n)
	# Endpoint weights should be halved to avoid aliasing
	cj = (numpy.mod(idx, n) == 0).choose(2., 1.)
	k = numpy.arange(1, int(n / 2) + 1)[:,numpy.newaxis]
	bk = (k < n / 2.).choose(1., 2.)
	# The weights are defined according to Waldvogel (2003)
	cos = numpy.cos(2. * k * nodes[numpy.newaxis,:])
	scl = bk / (4. * k**2 - 1.)
	weights = (cj / n) * (1. - numpy.sum(scl * cos, axis=0))
	return nodes, weights


def fejer2 (m):
	'''
	Compute the quadrature nodes and weights, in the interval [0, pi],
	using Fejer's second rule for an order m.
	'''
	n = m - 1
	idx = numpy.arange(m)
	nodes = idx * math.pi / float(n)
	k = numpy.arange(1, int(n / 2) + 1)[:,numpy.newaxis]
	sin = numpy.sin((2. * k - 1.) * nodes[numpy.newaxis,:])
	weights = (4. / n) * numpy.sin(nodes) * numpy.sum(sin / (2. * k - 1.), axis=0)
	return nodes, weights


def gaussleg (m, tol = 1e-9, itmax=100):
	'''
	Compute the Gauss-Legendre quadrature nodes and weights in the interval
	[0,pi] for a specified order m. The Newton-Raphson method is used to
	find the roots of Legendre polynomials (the nodes) with a maximum of
	itmax iterations and an error tolerance of tol.
	'''
	weights = numpy.zeros((m), dtype=numpy.float64)
	nodes = numpy.zeros((m), dtype=numpy.float64)

	nRoots = (m + 1) / 2

	for i in range(nRoots):
		# The initial guess is the (i+1) Chebyshev root
		t = math.cos (math.pi * (i + 0.75) / m)
		for j in range(itmax):
			# Grab the Legendre polynomial and its derviative
			p, dp = legendre (t, m)[:2]

			# Perform a Newton-Raphson update
			dt = -p/dp
			t += dt

			# Update the node and weight estimates
			nodes[i] = math.acos(t)
			nodes[-(i + 1)] = math.acos(-t)
			weights[i] = 2.0 / (1.0 - t**2) / (dp**2)
			weights[-(i + 1)] = weights[i]

			# Nothing left to do if tolerance was achieved
			if abs(dt) < tol: break

	return nodes, weights


def gausslob (m, tol = 1e-9, itmax=100):
	'''
	Compute the Gauss-Lobatto quadrature nodes and weights in the interval
	[0,pi] for a specified order m. The Newton-Raphson method is used to
	find the roots of derivatives of Legendre polynomials (the nodes) with
	a maximum of itmax iterations and an error tolerance of tol.
	'''
	weights = numpy.zeros((m), dtype=numpy.float64)
	nodes = numpy.zeros((m), dtype=numpy.float64)

	# This is the number of roots away from the endpoints
	nRoots = (m - 1) / 2

	# First compute the nodes and weights at the endpoints
	nodes[0] = 0.
	nodes[-1] = math.pi
	weights[0] = 2. / m / (m - 1.)
	weights[-1] = weights[0]

	for i in range(1, nRoots + 1):
		# The initial guess is halfway between subsequent Chebyshev roots
		t = 0.5 * sum(math.cos(math.pi * (i + j - 0.25) / m) for j in range(2));
		for j in range(itmax):
			# Grab the Legendre polynomial and its derviative
			p, dp, ddp = legendre (t, m - 1)

			# Perform a Newton-Raphson update
			dt = -dp / ddp
			t += dt

			# Update the node and weight estimates
			nodes[i] = math.acos(t)
			nodes[-(i + 1)] = math.acos(-t)
			weights[i] = 2. / m / (m - 1.) / (p**2)
			weights[-(i + 1)] = weights[i]

			# Nothing left to do if tolerance was achieved
			if abs(dt) < tol: break

	return nodes, weights


def complexmax (a):
	'''
	Compute the maximum element of a complex array like MATLAB does.
	'''
	# Return the maximum value of a numpy matrix.
	return a.flat[numpy.argmax(numpy.abs(a))]

def psnr (x, y):
	'''
	The peak SNR, in dB, of a matrix x relative to the matrix y.
	This assumes x = y + N, where N is noise and y is signal.
	'''
	# Compute the average per-pixel squared error
	err = numpy.sum (numpy.abs(x - y)**2) / float(prod(x.shape))
	# Compute the square of the maximum signal value
	maxval = numpy.max(numpy.abs(y))**2

	# Compute the peak SNR in dB
	return 10. * math.log10(maxval / err)

def mse (x, y):
	'''
	Report the mean squared error between the matrix x and the matrix y.
	'''
	err = numpy.sum (numpy.abs(x - y)**2)
	err /= numpy.sum (numpy.abs(y)**2)

	return math.sqrt(err)
