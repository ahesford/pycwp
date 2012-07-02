'''
Routines useful for wave physics computations, including CG-FFT, phase and
magnitude correction for comparison of k-space (FDTD) and integral equation
solvers, Green's functions, and a split-step solver.
'''

import sys, math, cmath, numpy as np, scipy as sp
from numpy import fft, linalg, random
import scipy.special as spec, scipy.sparse.linalg as la, scipy.sparse as sparse
from itertools import izip, product, repeat

from . import mio, cutil
from .f2py import pade, splitstep


def gencompress(c, rho, atn):
	'''
	Convert the sound speed c (m/s at zero frequency), the density rho
	(kg/m**3) and the attenuation slope atn (dB/cm/MHz) into two
	generalized compressibilities k = (k1, k2) used to define the sound
	speed according to Nachman, Smith and Waag (JASA, 1990).

	The time constants for the compressibilities k1 and k2 are,
	respectively, t1 = 11.49 ns and t2 = 77.29 ns.
	'''
	# Define polynomial coefficients for the curve fit
	a1 = (1.37424E-09, -7.86182E-09, 1.62596E-08, -1.23225E-08,
			5.42641E-06, -3.78429E-11, 2.76741E-10, -7.16265E-10,
			7.99459E-10, -3.46054E-10)

	a2 = (-6.85660E-09, 4.41817E-08, -9.61036E-08, 8.18166E-08,
			5.44780E-06, -3.47953E-12, 2.83491E-11, -8.23410E-11,
			1.04052E-10, -5.07517E-11)

	# Watch for zero attenuation
	alpha = (np.abs(atn) > 1e-6).choose(1., atn)
	asq = np.sqrt(alpha)

	# Compute the compressibilities
	k1, k2 = [((a[0] / c + a[5]) / alpha + (a[1] / c + a[6]) / asq +
			(a[2] / c + a[7]) + (a[3] / c + a[8]) * asq +
			(a[4] / c + a[9]) * alpha) / rho for a in [a1, a2]]

	# Correct the compressibilities where there is zero attenuation
	k1, k2 = [(np.abs(atn) > 1e-6).choose(0., k) for k in [k1, k2]]

	return k1, k2


def compress2spd(c, rho, k, f):
	'''
	Convert the sound speed c (m/s at zero frequency), the density rho
	(kg/m**3) and the generalized compressibilities k = (k1, k2) returned
	by the gencompress function into a sound speed at the specified
	frequency f (MHz).

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
	c = math.sqrt(2. / rho) / np.sqrt(kt + np.sqrt(kt**2 + ks**2))
	return c


def spd2ct(c, cbg, atn = None):
	'''
	Convert a wave speed profile c and an optional attenuation profile atn
	into a unitless, complex contrast profile relative to the background
	wave speed cbg.

	The speed units are mm / us; attenuation units are dB / cm / MHz.
	'''

	try:
		# The factor 0.1 converts dB / cm / MHz to dB / mm / MHz
		# The factor log(10) / 20 converts dB to Np
		scale = 0.1 * math.log(10) / 20
		# Multiplying atn by cbg converts it to dB per wavelength
		k = 2. * math.pi * cbg / c + 1j * cbg * scale * atn
	except TypeError: k = 2. * math.pi * cbg / c

	# Return the contrast profile
	return (k / 2. / math.pi)**2 - 1.


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
	# Compute and return the pattern
	return ctheta * np.exp(-a * np.sin(np.arccos(ctheta))**2)


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
	enum = lambda c: product(*[cv.flat for cv in c])

	# Compute the cell-relative quadrature points
	qpts = ([o + s * wts[i][0] for i, o, s in zip(c, src, sc)] for c in enum(coords))

	# Compute the corresponding quadrature weights
	qwts = (np.prod([wts[i][1] for i in c]) for c in enum(coords))

	# Sum all contributions to the integral
	ival = np.sum(w * ifunc(k, p, obs) for w, p in izip(qwts, qpts))

	return ival * np.prod(cell) / 2.**dim


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
		self.grf = greenfunc(k, r) * np.prod(cell)

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
		n = np.prod(obj.shape)

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

	def __init__(self, k0, nx, ny, h, l = 10, dz = None):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0. The wave
		is advanced in steps of dz or (if dz is not provided) h.

		A Hann window of thickness l is applied to each boundary of the
		field to attenuate the field in that region.
		'''

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0, self.l = h, k0, l
		# Set the step length
		self.dz = dz if dz else h

		# The default attenuation screen is None
		self._attenuator = None
		# The default incident field function is None
		self.setincident = None
		self.reset()


	def reset(self):
		'''
		Reset the propagating field and the index of refraction.
		'''
		# The default field to propagate is None
		self.fld = np.zeros(self.grid, dtype=np.complex64, order='F')
		# Initialize the index of refraction
		self.eta = [np.ones(self.grid, dtype=np.complex64, order='F')]


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


	def kxy(self):
		'''
		Cache and return the FFT-shifted values of kx and ky on the grid.
		'''
		try: return self._kxy
		except AttributeError:
			# Get the coordinate indices for each slab
			nx, ny = self.grid
			kx = 2. * math.pi * fft.fftfreq(nx, self.h)
			ky = 2. * math.pi * fft.fftfreq(ny, self.h)

			self._kxy = kx, ky
			return self._kxy


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.ogrid[:g[0], :g[1]]
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	def copyfield(self, fld = None):
		'''
		If fld is provided, copy the provided field to self.fld.
		Otherwise, return the contents of self.fld.
		'''
		if fld is None: return self.fld
		else: self.fld = fld


	def etaupdate(self, obj):
		'''
		Update the rolling buffer with the refractive index
		corresponding to the object contrast obj for the next slab.
		'''
		self.eta.append(splitstep.obj2eta(obj))
		efrac = splitstep.etafrac(*self.eta)
		self.eta.pop(0)

		return self.eta[0], efrac


	def advance(self, obj, bfld = None, prev = None, tau = None):
		'''
		Propagate a field through a slab with object contrast obj and
		use it to compute an estimate of the actual field in the slab.

		The field bfld, if provided, is the current guess of a field
		running counter to the field to be updated. The field prev, if
		provided, is the current guess of the field to be updated. The
		update is computed using a relaxation technique with parameter
		tau >= 2.

		If tau is None, forward-only propagation is assumed.
		'''
		# Convert the contrast to a scaled wave number
		eta, efrac = self.etaupdate(obj)

		# Propagate the field
		kx, ky = self.kxy()
		self.fld = splitstep.advance(self.fld, eta, self.k0, kx, ky, self.dz, self.l)

		if tau is None: return

		# If no backward-traveling or previous fields were provided,
		# assume they are zero
		if bfld is None: bfld = np.zeros(self.grid, dtype=obj.dtype, order='F')
		if prev is None: prev = np.zeros(self.grid, dtype=obj.dtype, order='F')

		# Apply a relaxation update to the traveling field
		self.fld = splitstep.update(self.fld, bfld, prev, efrac, tau)


class SplitPade(object):
	'''
	This class decomposes the split-step solution

		u(x, y, z + dz) = exp(1j * k * dz * sqrt(1 + Q)) u(x, y, z)

	for a differential operator Q into a split-step Padé solution

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
		Establish a split-step Padé approximant of desired order to the
		split-step propagation operator characterized by kz = k0 * dz
		for a propagation step dz through wave number k0.

		When the Padé approximant is used to compute propagation, a
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
		Return the value of the Padé approximant

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
		coefficients of the Padé approximant.
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
		iterations to compute the cofficients of the split-step Padé
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
		Detect whether more than one Padé denominator coefficients are
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
		scattering contrast obj using a split-step Padé approximant.

		The kwargs are passed to the solver to invert the denominator.
		'''
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
		shape = [np.prod(o.shape)]*2

		for i, (a, b, x0) in enumerate(zip(self.a, self.b, self.x0)):
			print 'Computing Padé contribution', i + 1
			mv = lambda x: flat(pade.scatop(o, grid(x), self.dz, self.k0, 1., b))
			A = la.LinearOperator(shape, dtype=complex, matvec=mv)

			x, info = solver(A, flat(f), x0=x0, **kwargs)
			if info != 0: print 'Iterative solver failed with error', info

			# Store the (advanced) previous solution for the next round
			self.x0[i] = x * np.exp(1j * self.k0 * self.dz)

			outfld += pade.scatop(o, grid(x), self.dz, self.k0, 0., a)

		return outfld
