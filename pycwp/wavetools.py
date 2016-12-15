'''
Routines useful for wave physics computations, including CG-FFT, phase and
magnitude correction for comparison of k-space (FDTD) and integral equation
solvers, Green's functions, and a split-step solver.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, math, cmath, numpy as np, scipy as sp
from numpy import fft, linalg, random
import scipy.special as spec, scipy.sparse.linalg as la, scipy.sparse as sparse
from itertools import izip, product as iproduct

from . import cutil

def crdtogrid(x, nx, dx=1.):
	'''
	Given physical coordinates x inside a grid with dimensions nx and
	spacing dx (maybe multidimensional), return the (possibly fractional)
	pixel coordinates assuming that the physical origin resides at the
	center of the grid.

	If any parameter is scalar, it is assumed to apply isotropically.
	'''
	# Check whether the coordinates are scalar
	try:
		len(x)
		scalar = False
	except TypeError: scalar = True

	# Check dimensionality of arguments and convert scalars
	x, nx, dx = cutil.matchdim(x, nx, dx)

	crd = tuple(xv / dxv + 0.5 * (nxv - 1) for xv, nxv, dxv in zip(x, nx, dx))
	if scalar: return crd[0]
	else: return crd


def gridtocrd(x, nx, dx=1.):
	'''
	Given pixel coordinates x (possibly fractional) inside a grid with
	dimensions nx and spacing dx (all of which may be multidimensional),
	return the physical coordinates assuming that the coordinate origin
	resides in the center of the grid.

	If any parameter is scalar, it is assumed to apply isotropically.
	'''
	# Check whether the coordinates are scalar
	try:
		len(x)
		scalar = False
	except TypeError: scalar = True

	# Check dimensionality of arguments and convert scalars
	x, nx, dx = cutil.matchdim(x, nx, dx)

	crd = tuple(dxv * (xv - 0.5 * (nxv - 1)) for xv, nxv, dxv in zip(x, nx, dx))
	if scalar: return crd[0]
	else: return crd


def gencompress(c, rho, atn):
	'''
	Convert the sound speed c (m/s at zero frequency), the density rho
	(kg/m**3) and the attenuation slope atn (dB/cm/MHz) into two
	generalized compressibilities k = (k1, k2) used to define the sound
	speed according to Nachman, Smith and Waag (JASA, 1990).

	The time constants for the compressibilities k1 and k2 are,
	respectively, t1 = 11.49 ns and t2 = 77.29 ns.

	Attenuation slopes less than 0.1 dB/cm/MHz are assumed to be zero.
	'''
	# Define polynomial coefficients for the curve fit
	a1 = (1.37424E-09, -7.86182E-09, 1.62596E-08, -1.23225E-08,
			5.42641E-06, -3.78429E-11, 2.76741E-10, -7.16265E-10,
			7.99459E-10, -3.46054E-10)

	a2 = (-6.85660E-09, 4.41817E-08, -9.61036E-08, 8.18166E-08,
			5.44780E-06, -3.47953E-12, 2.83491E-11, -8.23410E-11,
			1.04052E-10, -5.07517E-11)

	# Watch for zero attenuation
	alpha = (np.abs(atn) > 0.1).choose(1., atn)
	asq = np.sqrt(alpha)

	# Compute the compressibilities
	k1, k2 = [((a[0] / c + a[5]) / alpha + (a[1] / c + a[6]) / asq +
			(a[2] / c + a[7]) + (a[3] / c + a[8]) * asq +
			(a[4] / c + a[9]) * alpha) / rho for a in [a1, a2]]

	# Correct the compressibilities where there is zero attenuation
	k1, k2 = [(np.abs(atn) > 0.1).choose(0., k) for k in [k1, k2]]

	return k1, k2


def compress2spd(c, rho, k, f):
	'''
	Convert the sound speed c (m/s at zero frequency), the density rho
	(kg/m**3) and the generalized compressibilities k = (k1, k2) returned
	by the gencompress function into a sound speed at the specified
	frequency f (Hz).

	The time constants are fixed at t1 = 11.49 ns and t2 = 77.29 ns.
	'''
	# Compute the high-frequency asymptotic compressibility
	kinf = (1 / c**2 / rho) - np.sum(k, axis=0)
	# Compute the radian frequency and the fixed time constants
	w = 2 * math.pi * f
	tau = (11.49e-9, 77.29e-9)
	# Compute the first-order terms
	kt = kinf + np.sum((kv / (1. + (w * t)**2) for kv, t in zip(k, tau)), axis=0)
	# Compute the second-order corrections
	ks = np.sum((kv * t * w / (1. + (w * t)**2) for kv, t in zip(k, tau)), axis=0)
	# Return the sound speed at the desired frequency
	c = np.sqrt(2.) / np.sqrt(rho * (kt + np.sqrt(kt**2 + ks**2)))
	return c


def spd2ct(c, cbg, atn = None, rho = None):
	'''
	Convert a wave speed profile c, an optional attenuation profile atn and
	an optional density rho = (rho_r, L (rho_r)**(-1/2)), where rho_r is
	the relative density and L is the three-dimensional Laplacian, into a
	unitless, complex contrast profile relative to the background wave
	speed cbg.

	The speed units are mm / us; attenuation units are dB / cm / MHz.
	'''

	if atn is not None:
		# The factor 0.1 converts dB / cm / MHz to dB / mm / MHz
		# The factor log(10) / 20 converts dB to Np
		scale = 0.1 * math.log(10) / 20
		# Multiplying atn by cbg converts it to dB per wavelength
		k = 2. * math.pi * cbg / c + 1j * cbg * scale * atn
	else: k = 2. * math.pi * cbg / c

	# Compute the contrast profile
	obj = (k / 2. / math.pi)**2 - 1.
	# Add density variations if provided
	if rho is not None: obj -= np.sqrt(rho[0]) * rho[1] / (2. * math.pi)**2

	return obj


def directivity(obs, src, d, a):
	'''
	Evaluate a directivity pattern cos(theta) exp(-a sin(theta)**2), where
	theta is the angle between a direction d and the distance r = (obs -
	src), where obs and src are the respective positions of the observer
	and source.
	'''
	# Compute the distance r and its norm
	r = [x - y for x, y in izip(obs, src)]
	# Compute the norms of the distance and focal axis
	rn = np.sqrt(reduce(np.add, [rv**2 for rv in r]))
	dn = np.sqrt(reduce(np.add, [dv**2 for dv in d]))

	# Compute the cosine of the angle
	ctheta = reduce(np.add, [x * y for x, y in izip(r, d)]) / rn / dn
	# Compute the square of the sine of the angle
	sthsq = 1. - ctheta**2
	# Compute and return the pattern
	return ctheta * np.exp(-a * sthsq)


def green3d(k, r):
	'''
	Evaluate the 3-D Green's function for a distance r and wave number k.
	'''
	return np.exp(1j * k * r) / (4. * math.pi * r)


def green2d(k, r):
	'''
	Evaluate the 2-D Green's function for a distance r and wave number k.
	'''
	return 0.25j * (spec.j0(k * r) + 1j * spec.y0(k * r))


def green3dpar(k, r, z):
	'''
	Return the 3-D Green's function for the parabolic wave equation for a
	total distance r, wave number k, and a distance z along the propagation
	axis. This formulation omits the exp(-ikz) factor present in Levy
	(2000) that converts the wave field into a slowly-varying function in
	z. Consequently, steps in z should be small with respect to wavelength.
	'''
	ikr = 1j * k * r
	return np.exp(ikr) * (1. - ikr) * z / (2. * math.pi * r**3)


def green2dpar(k, r, z):
	'''
	Return the 2-D Green's function for the parabolic wave equation for a
	total distance r, wave number k, and a distance z along the propagation
	axis. This formulation omits the exp(-ikz) factor present in Levy
	(2000) that converts the wave field into a slowly-varying function in
	z. Consequently, steps in z should be small with respect to wavelength.
	'''
	h1 = spec.j1(k * r) + 1j * spec.y1(k * r)
	return 0.5j * k * z * h1 / r


def greenduf3d(k, x, u, v):
	'''
	Evaluate the Green's function in Duffy-transformed coordinates.

	The origin is assumed to be the observation point.
	'''
	sq = np.sqrt(1. + u**2 + v**2)
	return x * np.exp(1j * k * x * sq) / (4. * math.pi * sq)


def duffyint(k, obs, cell, n = 4, greenfunc = greenduf3d):
	'''
	Evaluate the self-integration of order n of the smoothed Green's
	function over a cubic cell, with length dc, using Duffy's
	transformation.
	'''

	dim = len(obs)
	if dim != len(cell): raise ValueError('Dimension of obs and cell must agree.')

	# Wrap the Duffy Green's function to the form expected by srcint
	grf = lambda kv, s, o: greenfunc(k, *s)

	# Make sure that the obs and cell iterables are lists
	if not isinstance(obs, list): obs = list(obs)
	if not isinstance(cell, list): cell = list(cell)

	# Define a generic function to build the cell size and center
	def duffcell(s, d, sgn):
		sgn = 1 if sgn > 0 else -1
		l = 0.5 * d[0] - sgn * s[0]
		dc = [l] + [dv / l for dv in d[1:]]
		src = [0.5 * l] + [-sv / l for sv in s[1:]]
		return src, dc

	# Store the final integration value
	val = 0.

	# Deal with each axis in succession
	for i in range(dim):
		# Integrate over the pyramid along the +x axis
		src, dc = duffcell(obs, cell, 1)
		# The obs argument is ignored so it is set to 0
		val += srcint(k, src, [0.]*dim, dc, grf, n)

		# Integrate over the pyramid along the -x axis
		src, dc = duffcell(obs, cell, -1)
		# The obs argument is ignored so it is set to 0
		val += srcint(k, src, [0.]*dim, dc, grf, n)

		# Rotate the next axis into the x position
		obs = cutil.rotate(obs)
		cell = cutil.rotate(cell)

	return val


def srcint(k, src, obs, cell, ifunc, n = 4, wts = None):
	'''
	Evaluate source integration, of order n, of the pairwise Green's
	function for wave number k from source location src to observer
	location obs. The list cell specifies the dimensions of the cell.

	The pairwise Green's function function ifunc takes arguments (k, s, o),
	where k is the wave number, s is the source location and o is the
	observer location. The source position s varies throughout the cell
	centered at src according to Gauss-Legendre quadrature rules.

	If specified, wts should be an n-by-2 array (or list of lists) in which
	the first column contains the quadrature points and the second column
	contains the corresponding quadrature weights.
	'''

	dim = len(src)

	if len(obs) != dim: raise ValueError('Dimension of src and obs must agree.')
	if len(cell) != dim: raise ValueError('Dimension of src and cell must agree.')

	# Compute the node scaling factor
	sc = [0.5 * c for c in cell]

	# Grab the roots and weights of the Legendre polynomial of order n
	if wts is None: wts = spec.legendre(n).weights

	# Compute a sparse coordinate grid for sampling within the cell
	coords = np.ogrid[[slice(n) for i in range(dim)]]
	# Create an iterator factory to loop through coordinate pairs
	enum = lambda c: iproduct(*[cv.flat for cv in c])

	# Compute the cell-relative quadrature points
	qpts = ([o + s * wts[i][0] for i, o, s in zip(c, src, sc)] for c in enum(coords))

	# Compute the corresponding quadrature weights
	qwts = (cutil.prod(wts[i][1] for i in c) for c in enum(coords))

	# Sum all contributions to the integral
	ival = np.sum(w * ifunc(k, p, obs) for w, p in izip(qwts, qpts))

	return ival * cutil.prod(cell) / 2.**dim


class CGFFT(object):
	def __init__(self, k, grid, cell, greenfunc = green3d):
		'''
		Compute the extended Green's function with wave number k for
		use in CG-FFT over a non-extended grid with dimensions
		specified in the list grid. Each cell has dimensions specified
		in the list cell.

		The Green's function greenfunc(k,r) takes as arguments the wave
		number and a scalar distance or numpy array of distances
		between the source and observation locations. The 3-D Green's
		function is used by default.
		'''

		dim = len(grid)

		if dim != len(cell):
			raise ValueError('Dimensionality of cell and grid lists must agree.')

		# Build the coordinate index arrays in floating-point
		coords = np.ogrid[[slice(2 * g) for g in grid]]

		# Wrap the coordinate arrays and convert to cell coordinates
		coords = [c * (x < g).choose(x - 2 * g, x)
				for c, g, x in zip(cell, grid, coords)]

		# Compute the distances along the coordinate grids
		r = np.sqrt(np.sum([c**2 for c in coords], axis=0))

		# Evaluate the Green's function on the grid
		self.grf = greenfunc(k, r) * cutil.prod(cell)

		# Correct the zero value to remove the singularity
		self.grf[[slice(1) for d in range(dim)]] = duffyint(k, [0.]*dim, cell)

		# Compute the FFT of the extended-grid Green's function
		self.grf = fft.fftn(k**2 * self.grf)


	def applygrf(self, fld):
		'''
		Apply the Green's function grf to the field fld using FFT convolution.
		'''
		if len(fld.shape) != len(self.grf.shape):
			raise ValueError('Arguments must have same dimensionality.')

		# Compute the convolution with the Green's function
		efld = fft.ifftn(self.grf * fft.fftn(fld, s = self.grf.shape))

		# Return the relevant portion of the solution
		sl = [slice(None, s) for s in fld.shape]
		return efld[sl]


	def scatmvp(self, fld, obj):
		'''
		Apply the scattering operator to an input field using FFT convolution.
		'''
		return fld - self.applygrf(fld * obj)


	def solve(itfunc, grf, obj, rhs, **kwargs):
		'''
		Solve the scattering problem for a precomputed Green's function
		grf, an object contrast (potential) obj, and an incident field
		rhs. The iterative solver itfunc is used, from
		scipy.sparse.linalg.
		'''

		# Compute the dimensions of the linear system
		n = cutil.prod(obj.shape)

		# Function to compute the matrix-vector product
		mvp = lambda v: self.scatmvp(v.reshape(obj.shape, order='F'), obj).ravel('F')

		# Build the scattering operator class
		scatop = la.LinearOperator((n,n), matvec=mvp, dtype=rhs.dtype)

		# Solve the scattering problem
		p, info = itfunc(scatop, rhs.flatten('F'), **kwargs)
		p = p.reshape(obj.shape, order='F')
		return p, info


def kcomp(inc, src, dx):
	'''
	Compute the phase shift and magnitude scaling factors, relative to an
	integral equation solution with negative time convention, for an
	incident field computed as a frequency component of an FDTD solution
	using the positive time convention. Returns (mag, phase) such that the
	total fields between the IE and FDTD solutions are related by

		IE = mag * abs(FDTD) * exp(1j * (phase - angle(FDTD))).

	The incident field is taken as a z = 0 slab defined on a grid, and the
	source location should be specified using grid coordinates. The grid
	spacing, dx (in wavelengths), is uniform in the x, y and z coordinates.

	The reference point in the incident slab is the (x,y) project of the
	source location. Thus, for src = [i, j, k], the value of the Green's
	function for a distance k * dx is compared to the value of the incident
	field at point inc[i,j].
	'''

	# Compute the value of the Green's function at the point of projection
	# onto the plane
	gval = green3d (2 * math.pi, dx * src[-1])

	# Compute the magnitude scaling
	mag = np.abs(gval / inc[src[0]][src[1]])

	# Compute the phase shift
	delta = cmath.phase(gval) + cmath.phase(inc[src[0]][src[1]])

	return mag, delta


def kscale(inc, scat, mag, delta):
	'''
	Scale (with contant mag), phase-shift (with offset delta), and combine
	the incident and scattered field (inc and scat, respectively) produced
	using a k-space (FDTD) method with positive time convention to produce
	a total field to compare with an integral equation solution using the
	negative time convention.
	'''

	return mag * np.abs(inc + scat) * np.exp(1j * (delta - np.angle(inc + scat)))
