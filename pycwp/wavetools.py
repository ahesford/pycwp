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


class SplitStep(object):
	'''
	A class to advance a wavefront through an arbitrary medium according to
	the split-step method.
	'''

	def __init__(self, k0, nx, ny, h, l = 10, dz = None, w = 0.32):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0. The wave
		is advanced in steps of dz or (if dz is not provided) h.

		A Hann window of thickness l is applied to each boundary of the
		field to attenuate the field in that region.

		The parameter w (as a multiplier 1 / w**2) governs high-order
		spectral corrections.
		'''
		from .ftntools import splitstep

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0, self.l = h, k0, l
		self.w = np.float32(w)
		# Set the step length
		self.dz = dz if dz else h

		# The default attenuation screen is None
		self._attenuator = None
		# The default incident field function is None
		self.setincident = None
		self.reset()

		# Pre-plan FFTs for efficiency
		splitstep.fft.plan(nx, ny)


	def reset(self):
		'''
		Reset the propagating field and the index of refraction.
		'''
		# The default field to propagate is None
		self.fld = np.zeros(self.grid, dtype=np.complex64, order='F')
		# Initialize the index of refraction
		self.eta = np.ones(self.grid, dtype=np.complex64, order='F')


	def setincgen(self, func):
		'''
		Given a function func(obs), where obs is an (x,y,z) triple of
		coordinates in an observer plane, assign to self.setincident a
		function of the form f(z) that takes the height z of the slab
		and sets in self.fld the incident field in the slab.
		'''
		def f(z):
			x, y = self.slicecoords()
			self.fld = func((x, y, z),)
		self.setincident = f


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.ogrid[:g[0], :g[1]]
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	def etaupdate(self, obj):
		'''
		Update the rolling buffer with the refractive index
		corresponding to the object contrast obj for the next slab.
		Return the current refrective indices and the reflection
		coefficients for the slab.
		'''
		from .ftntools import splitstep
		# Grab the current index of refraction
		cur = self.eta
		# Store the next index of refraction
		self.eta = splitstep.obj2eta(obj)
		# Return the current and next slabs
		return cur, self.eta


	def advance(self, obj, bfld = None, shift = False):
		'''
		Propagate a field through the current slab and transmit it
		through an interface with the next slab characterized by object
		contrast obj.

		The field bfld, if provided, runs opposite the direction of
		propagation and will be reflected and added to the propagated
		field after transmission across the interface.

		If shift is True, the forward and backward fields are also
		shifted by half a slab to agree with full-wave solutions.
		'''
		from .ftntools import splitstep

		# Grab the current and next indices of refraction
		eta, enxt = self.etaupdate(obj)

		# For half-plane shifts, compute the field to be shifted
		if shift:
			if bfld is None: midplane = self.fld.copy()
			else: midplane = self.fld + bfld

		# Apply a Hann window to the field if desired
		if self.l > 0: self.fld = splitstep.hann(self.fld, self.l)

		# Wide-angle propagation of the field through the slab
		self.fld = splitstep.advance(self.fld, eta,
				self.k0, self.h, self.dz, self.w)

		# Compute the reflection coefficients for the slab
		rc = splitstep.rcoeff(eta, enxt)
		# Compute the transmission and reflection from the interface
		if bfld is None: self.fld = splitstep.transmit(self.fld, rc)
		else: self.fld = splitstep.txreflect(self.fld, bfld, rc)

		# If no shifting is desired, return the transmitted field
		if not shift: return self.fld

		# Otherwise, apply the Hann window and propagate a half step
		if self.l > 0: midplane = splitstep.hann(midplane, self.l)
		midplane = splitstep.advance(midplane, eta,
				self.k0, self.h, 0.5 * self.dz, self.w)
		return midplane


class SplitPade(object):
	'''
	This class decomposes the split-step solution

		u(x, y, z + dz) = exp(1j * k * dz * sqrt(1 + Q)) u(x, y, z)

	for a differential operator Q into a split-step Pade solution

		u(x, y, z + dz) = e + sum(inv(1 + b[l] * Q) * a[l] * Q * u(x, y, z)),

	where e = exp(1j * k * dz), for l from 1 to some order N and a[l]
	and b[l] are coefficients subject to the constraint

		sum(a[l] / b[l]) = -1.

	In other words,

		exp(1j * k * dz * sqrt(1 + Q)) = e + sum(a[l] * Q / (1 + b[l] * Q))

	approximately. The class uses a modified Newton method to find the
	coefficients a[l] and b[l] for a prescribed order.
	'''
	def __init__(self, k0, dz, order = 8, l = None):
		'''
		Establish a split-step Pade approximant of desired order to the
		split-step propagation operator characterized by kz = k0 * dz
		for a propagation step dz through wave number k0.

		When the Pade approximant is used to compute propagation, a
		Hann window of length l (if provided) is applied to the field
		to avoid aliasing problems.
		'''
		self.k0 = k0
		self.dz = dz
		self.ikz = 1j * k0 * dz
		self.eikz = np.exp(self.ikz)
		self.order = order

		# Initialize the coefficient vectors
		self.b = np.linspace(1e-2, 1., self.order, False).astype(complex)
		self.a = random.rand(self.order,).astype(complex)

		# Initialize the derivative coefficients of the propagator
		self.derivs = self.operderivs()

		# Build the two-sided Hann window for apodization
		if l is not None and l > 0: self.hann = np.hanning(2 * l)
		else: self.hann = None

		# Initial guesses
		self.x0 = [None]*self.order


	@staticmethod
	def sspcoeffs(p, scale = 1.):
		'''
		Given a series of quasi-polynomial coefficients in the
		dictionary p, return in np the coefficients for the derivative
		of the operator

			f(Q) = exp(sqrt(1 + Q)) * sum(p[i] * sqrt(1 + Q)**i)

		such that

			f'(Q) = exp(sqrt(1 + Q)) * sum(np[i] * sqrt(1 + Q)**i).

		The keys of p and np represent the powers of sqrt(1 + Q), while
		the values are the coefficients.

		Each coefficient will be multiplied by scale, if provided.
		'''
		# A factor of 0.5 appears as a result of the derivative
		mpy = 0.5 * scale
		# Drop the power of each coefficient by one
		shifted = [(s - 1, mpy * c) for s, c in p.iteritems()]
		# Take the derivative of each term and drop the powers by one
		shderiv = [(s - 2, mpy * c * s) for s, c in p.iteritems() if s != 0]

		np = dict(shifted)
		for s, c in shderiv:
			try: np[s] += c
			except KeyError: np[s] = c

		return np


	def operderivs(self, q = 0):
		'''
		Return, in an array, the value of the propagation operator

			exp(1j * k0 * z * sqrt(1 + q))

		and its Taylor coefficients to order 2 * self.order - 1.
		'''
		# Initialize the quasi-polynomial for the propagator
		p = dict(((0, 1), ))
		# Compute the function value
		vals = [np.exp(self.ikz * np.sqrt(1. + q))]
		# A function to compute each term in the derivative sums
		qsq = np.sqrt(1. + q)
		term = lambda n, j: (self.ikz)**(n - j) * p[-n - j] * qsq**(-n - j)
		# Compute each of the derivatives
		for n in range(1, 2 * self.order):
			# Scaling by 1/n yields Taylor coefficients instead of
			# raw derivatives to avoid overflow
			p = self.sspcoeffs(p, 1. / n)
			# Compute the derivative as the sum of its terms
			vals.append(vals[0] * np.sum([term(n, j) for j in range(n)]))

		return np.array(vals)


	def padederivs(self, q = 0):
		'''
		Return the value of the Pade approximant

			exp(1j * k0 * dz) + sum(a[l] * q / (1 + b[l] * q))

		and its Taylor coefficients to order 2 * self.order - 1.
		'''
		# Compute the function value
		vals = [self.eikz + np.sum(self.a * q / (1. + self.b * q))]
		# Compute each of the derivatives
		a, b = self.a, self.b
		for n in range(1, 2 * self.order):
			d = np.sum(a * (-b)**(n - 1) / (1. + b * q)**(n + 1))
			vals.append(d)

		return np.array(vals)


	def costfunc(self):
		'''
		Return the value of the cost functional that constrains the
		coefficients of the Pade approximant.
		'''
		cost = self.padederivs() - self.derivs
		cost[0] = np.sum(self.a / self.b) + self.eikz
		return cost


	def jacobian(self):
		'''
		Return the Jacobian of the cost functional.
		'''
		jac = np.empty([2 * self.order] * 2, dtype=complex)

		# Build the Taylor-matching rows of the Jacobian
		for n in range(1, 2 * self.order):
			jac[n, :self.order] = (-self.b)**(n - 1)
			jac[n, self.order:] = -(n - 1) * self.a * (-self.b)**(n - 2)

		# Build the constraint (stability) row of the Jacobian
		jac[0, :self.order] = 1. / self.b
		jac[0, self.order:] = -self.a / self.b**2

		return jac


	def newton(self, maxiter = 100, epsilon = 1., tol = 1e-6):
		'''
		Run a modified Newton method with a maximum of maxiter
		iterations to compute the cofficients of the split-step Pade
		approximation to the propagation operator.

		The parameter epsilon may be less than 1 to stabilize the
		convergence behavior.

		Iteration will terminate early if the magnitude of the update,
		relative to the existing solution, falls below tol.
		'''
		for i in range(maxiter):
			# Compute the Jacobian at the current point
			j = self.jacobian()
			# Solve for the next update
			v = linalg.lstsq(j, self.costfunc())[0]
			# Copy the updated solution into the class parameters
			self.a -= epsilon * v[:self.order]
			self.b -= epsilon * v[self.order:]
			# Check for degeneracy
			self.stabilize(tol)

			if linalg.norm(self.costfunc()) < tol:
				print 'Convergence after %d iterations' % i
				break


	def stabilize(self, tol = 1e-6):
		'''
		Detect whether more than one Pade denominator coefficients are
		close within tolerance tol (a sign of degeneration to
		lower-order approximants) and change one (and its corresponding
		numerator) randomly.

		Also detect denominator coefficients with magnitudes greater
		than unity (which can instability or degeneracy) and numerator
		coefficients with magnitudes below the tolerance tol (another
		sign of degeneracy) and replace the corresponding numerator,
		denominator pair with randomly selected values.
		'''
		# Sort the numerator and denominator pairs by denominator
		c = [(a, b) for a, b in zip(self.a, self.b)]
		c.sort(key = lambda e: (e[1].real, e[1].imag))

		# Seed the first pair
		uc = [c[0]]
		for cv in c[1:]:
			# Skip current values too close to the last unique value
			if abs(cv[-1] - uc[-1][1]) / abs(uc[-1][1]) < tol: continue
			# Otherwise, update the last unique value
			uc.append(cv)

		# Remove large denominators and small numerators
		uc = [e for e in uc if abs(e[1]) <= 1. and abs(e[0]) > tol]

		# Replace skipped pairs with random values
		for i in range(self.order - len(uc)):
			uc.append((random.random(), random.random()))

		# Store the new coefficients in the original arrays
		self.a[:] = [e[0] for e in uc]
		self.b[:] = [e[1] for e in uc]


	def propagate(self, fld, obj, solver=la.lgmres, **kwargs):
		'''
		Compute the propagation of a field fld through a slab with
		scattering contrast obj using a split-step Pade approximant.

		The kwargs are passed to the solver to invert the denominator.
		'''
		from .ftntools import pade

		# Attenuate the field before propagation
		if self.hann is not None:
			l = len(self.hann) / 2
			fld[:l,:] *= self.hann[:l, np.newaxis]
			fld[-l:,:] *= self.hann[-l:, np.newaxis]
			fld[:,:l] *= self.hann[np.newaxis, :l]
			fld[:,-l:] *= self.hann[np.newaxis, -l:]

		# Ensure the field and contrast are 128-bit complex values
		f = fld.astype(complex)
		o = obj.astype(complex)

		# Define functions to convert between gridded and flattened arrays
		grid = lambda x: x.reshape(f.shape, order='F')
		flat = lambda x: x.ravel('F')

		# Compute the first solution term and the RHS for linear systems
		outfld = self.eikz * f

		# Build the shape of the flattened matrix
		shape = [cutil.prod(o.shape)]*2

		for i, (a, b, x0) in enumerate(zip(self.a, self.b, self.x0)):
			print 'Computing Pade contribution', i + 1
			mv = lambda x: flat(pade.scatop(o, grid(x), self.dz, self.k0, 1., b))
			A = la.LinearOperator(shape, dtype=complex, matvec=mv)

			x, info = solver(A, flat(f), x0=x0, **kwargs)
			if info != 0: print 'Iterative solver failed with error', info

			# Store the (advanced) previous solution for the next round
			self.x0[i] = x * np.exp(1j * self.k0 * self.dz)

			outfld += pade.scatop(o, grid(x), self.dz, self.k0, 0., a)

		return outfld


class EikonalSweep(object):
	'''
	A class to represent a 3-D solution to the Eikonal equation on a
	regular grid (as a pycwp.boxer.Box3D object) using the fast sweeping
	method in Zhao, "A fast sweeping method for Eikonal equations", Math.
	Comp. 74 (2005), 603--627.
	'''
	def __init__(self, box):
		'''
		Initialize the fast sweeping method on a pycwp.boxer.Box3D grid
		box. The sweep object captures a reference to the box rather
		than copying the box altogether.
		'''
		if any(ax < 2 for ax in box.ncell):
			raise ValueError('Cell count in each dimension must exceed 1')
		self.box = box


	def _neighbor(self, t, pt, axis):
		'''
		Given a Numpy array t of shape self.box.ncell (the shape and
		type of which are not checked), return the smaller of the
		values that staddle index pt = (i, j, k) along the given axis.

		If one neighbor value does not exist (because the point is on
		an edge of the grid), the other neighbor is always returned.
		'''
		pt = tuple(pt)

		lpt = pt[:axis] + (pt[axis] - 1,) + pt[axis+1:]
		rpt = pt[:axis] + (pt[axis] + 1,) + pt[axis+1:]

		if rpt[axis] >= self.box.ncell[axis]: return t[lpt]
		if lpt[axis] < 0: return t[rpt]

		return min(t[lpt], t[rpt])


	def sweep(self, t, s, octant, inplace=False):
		'''
		Perform one fast sweep, with axial directions determined by the
		quadrant argument (which must satisfy 0 <= octant < 8), to
		update the Eikonal t for a slowness (inverse speed) s.

		The return value is the tuple (nt, count), where nt is the
		updated Eikonal and count is the number of modified grid
		values; if count is 0, no values were changed in the sweep.

		Both t and s must be 3-D Numpy arrays with a shape that matches
		self.box.ncell, or compatible sequences that will be converted
		as necessary.

		The sweep uses a Godunov upwind difference scheme with
		one-sided differences at the boundary. The directions of sweeps
		along each axis are determined by decomposing the octant
		bitfield (1 <= octant <= 8) into (x, y, z) bit values according
		to the relationship

			octant = (z << 2) & (y << 1) & x + 1

		when 1 <= octant <= 8. The directions of the axial sweeps are
		increasing (decreasing) when their respective bit values are 0
		(1).

		If inplace is True, the solution will be updated in place if
		the input t is already a suitable array (i.e., the output of
		asarray(t) is either t or a view on t). In this case, the first
		return value of this method will be identical to t. The value
		of inplace is ignored if asarray(t) creates a new array.
		'''
		if not 1 <= octant <= 9:
			raise ValueError('Argument "octant" must satisfy 1 <= octant <= 8')

		# Make sure s and t are properly shaped arrays
		s = np.asarray(s)

		grid = self.box.ncell

		if grid != s.shape:
			raise TypeError('Argument "s" must have shape %s' % (grid,))

		# Always copy t if inplace is unwanted
		if inplace: t = np.asarray(t)
		else: t = np.array(t)

		if grid != t.shape:
			raise TypeError('Argument "t" must have shape %s' % (grid,))

		# Offset octant to get a "zyx" bitfield, where the bit for
		# each coordinate axis is 0 for forward and 1 for backward
		octant -= 1

		# Build the iterators for each axis
		axiter = []
		for nv in grid:
			# Check the direction of the current axis
			if not (octant & 0x1): axi = xrange(nv)
			else: axi = xrange(nv - 1, -1, -1)
			axiter.append(axi)
			# Shift octant to pull the next axis
			octant >>= 1

		# Find scale factors for nonuniform grids
		h = self.box.cell
		hm = min(h)
		h = tuple(hv / hm for hv in h)

		# Perform the sweep and track updated values
		updates = 0
		for pt in iproduct(*axiter):
			# Pull contributions along each axis, ordered by value
			# Also note the step size along each axis
			(a, b, c), (ha, hb, hc) = zip(*sorted(
				(self._neighbor(t, pt, a), h[a]) for a in xrange(3)))

			# First try a single-axis solution
			fh = s[pt] * hm
			nt = a + fh * ha

			# If solution is too large, consider two axes
			if nt > b:
				# RHS and grid scales are only used squared
				fh = fh**2
				ha, hb = ha**2, hb**2
				# Find terms in quadratic formula
				Aq = ha + hb
				Bq = hb * a + ha * b
				Cq = hb * a**2 + ha * (b**2 - fh * hb)

				nt = (Bq + math.sqrt(Bq**2 - Aq * Cq)) / Aq

				# If solution is too large, need all three axes
				if nt > c:
					# Final axis scale was not previously squared
					hc = hc**2
					# Find terms in quadratic formula
					Aq = hb * hc + ha * hc + ha * hb
					Bq = hb * hc * a + ha * (hc * b + hb * c)
					Cq = (hb * hc * (a**2 - fh * ha) +
						ha * (hc * b**2 + hb * c**2))

					nt = (Bq + math.sqrt(Bq**2 - Aq * Cq)) / Aq

			if nt < t[pt]:
				updates += 1
				t[pt] = nt

		return t, updates


	def gauss(self, s, src, report=False):
		'''
		Given a slowness s, a 3-D Numpy array with shape self.box.ncell
		or an array-compatible sequence, and a source location src in
		natural coordinates, compute and return the Eikonal t (with the
		same shape as s) that describes arrival times from the source
		to every point in the domain according to the slowness map.
		Gauss-Seidel iterations with alternating sweep directions are
		applied until no updates are made to the arrival-time map.

		For the purposes of initialization, boundary cells are assumed
		to extend outward to infinity to provide "background" slowness
		when a source falls outside of the grid.

		If report is True, the total number of updated grid points will
		be printed after each complete round of alternating sweeps.
		'''
		box = self.box
		ncell = box.ncell

		s = np.asarray(s)
		if s.shape != ncell or not np.issubdtype(s.dtype, np.floating):
			raise TypeError('Array "s" must be real with shape %s' % (ncell,))

		src = np.asarray(src)
		if src.ndim != 1 or len(src) != 3:
			raise TypeError('Array "src" must be a three-element sequence')

		# Find a (relaxed) upper bound on propagation distance
		# First compute the distance from source to domain center
		dmax = linalg.norm(src - box.midpoint)
		# Now add the diameter of the sphere that encloses the domain
		dmax += linalg.norm(box.length)

		# Initialize arrival times to beyond-possible values
		t = np.empty_like(s)
		t[:,:,:] = np.max(s) * dmax

		# Convert source to grid coords and find (extended) enclosing cell
		src = np.array(box.cart2cell(*src))
		sgrd = np.clip(src.astype(np.int64), (0,)*3, [n - 1 for n in ncell])

		slw = s[tuple(sgrd)]
		h = box.cell

		# Assign arrival-time values to grid points surrounding cell
		for inc in iproduct(xrange(2), repeat=3):
			# Check that the grid point is valid
			npt = tuple(sgrd + inc)
			if any(x >= n for x, n in izip(npt, box.ncell)): continue
			t[npt] = slw * linalg.norm((src - npt) * h)

		it = 0
		while True:
			updates = 0
			for octant in xrange(1, 9):
				t, upc = self.sweep(t, s, octant, True)
				if report:
					print 'Iteration %d, octant %d: %d updates' % (it, octant, upc)
				updates += upc
			it += 1
			if not updates: break

		return t
