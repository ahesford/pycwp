'''
Routines useful for wave physics computations, including CG-FFT, phase and 
magnitude correction for comparison of k-space (FDTD) and integral equation
solvers, Green's functions, and a split-step solver.
'''

import math, cmath, numpy as np, scipy as sp
import numpy.fft as fft, scipy.special as spec, scipy.sparse.linalg as la
from numpy.linalg import norm
from itertools import izip, product

from .cutil import rotate
from . import mio

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
		obs = rotate(obs)
		cell = rotate(cell)

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

	def __init__(self, k0, nx, ny, h, dz = None):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0. The wave
		is advanced in steps of dz or (if dz is not provided) h.
		'''

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0 = h, k0
		# Set the step length
		self.dz = dz if dz else h

		# The default attenuation screen is None
		self._attenuator = None
		# Build the Green's function for the stepper
		self.buildgreen()


	def kxy(self):
		'''
		Cache and return the FFT-shifted values of kx and ky on the grid.
		'''
		try: return self._kxy
		except AttributeError:
			# Get the coordinate indices for each slab
			nx, ny = self.grid
			sl = np.ogrid[-nx/2:nx/2, -ny/2:ny/2]

			# Compute the transverse wave numbers
			dk = 2. * math.pi / self.h
			kx, ky = [fft.fftshift((dk * s) / n) 
					for s, n in zip(sl, [nx, ny])]

			self._kxy = kx, ky
			return self._kxy


	def buildgreen(self, dz = None):
		'''
		Build the Green's for propagation through a step dz. If dz is
		not provided, use the default specified for the class instance.

		The exact spectral representation of the Green's function is
		used. This seems to yield some error relative to the FFT of the
		spatial representation computed with green3dpar.

		For a fully accurate spatial convolution, the Green's function
		should be evaluated on a grid twice as large in each dimension
		as the field samples. However, since the spectral Green's
		function decays exponentially on the extended grid, sufficient
		accuracy seems to be obtained without extending the grid.
		'''
		if dz is not None: self.dz = dz

		kx, ky = self.kxy()
		kz = np.sqrt(complex(self.k0**2) - kx**2 - ky**2)
		self.grf = np.exp(1j * self.dz * kz)


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.ogrid[:g[0], :g[1]]
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	@property
	def attenuator(self):
		'''
		Return a screen to attenuate the field at each slab edge. If
		one was not previously set, a screen that does not attenuate is
		created.
		'''
		return self._attenuator


	@attenuator.setter
	def attenuator(self, l):
		'''
		Generate attenuate screens using the Hann function with the
		specified width l.
		'''
		def hann(t, l): return np.sin(math.pi * t / (2 * l - 1.))**2
		nx, ny = self.grid
		# Create the left half of the window along each axis
		tx, ty = np.arange(nx), np.arange(ny)
		wx, wy = [(t < l).choose(1., hann(t, l)) for t in [tx, ty]]
		# Multiply by the reverse to symmetrize the window
		wx, wy = [w * w[::-1] for w in [wx, wy]]

		# Store the attenuation screens in a broadcastable form
		self._attenuator = (wx[:,np.newaxis], wy[np.newaxis,:])


	@attenuator.deleter
	def attenuator(self): del self._attenuator


	def phasescreen(self, eta):
		'''
		Generate the phase screen for a slab characterized by scaled
		wave number eta.
		'''
		return np.exp(1j * self.dz * self.k0 * (eta - 1.))


	def wideangle(self, lap, eta):
		'''
		Return the wide-angle corrections to a field with Laplacian lap
		propagating through a medium slab with index of refraction eta.
		'''
		cor = 1j * self.k0 * self.dz * (eta - 1.) / (2. * eta)
		return cor * lap


	def advance(self, fld, obj):
		'''
		Use the spectral propagator, the phase screen for the current
		slab, and the attenuation screen to advance the field through
		one layer of a specified medium with contrast obj. The
		attenuation screen must have previously been established.
		'''
		# Convert the contrast to a scaled wave number
		eta = np.sqrt(obj + 1.)

		# Attenuate the field if desired
		if self.attenuator is not None:
			fld = (fld * self.attenuator[0]) * self.attenuator[1]

		# Compute the extended Green's-function convolution in spectrum
		efld = self.grf * fft.fftn(fld, s=self.grf.shape)

		# Prepare the domain truncation slice objects
		sl = [slice(g) for g in self.grid]

		# Pre-compute the Laplacian of the field
		kx, ky = self.kxy()
		lap = fft.ifftn((kx**2 * efld + ky**2 * efld) / self.k0**2)[sl]
		# Return the field to the spatial domain with wide-angle corrections
		fld = fft.ifftn(efld)[sl] + self.wideangle(lap, eta)
		# Apply the phase screen as the final step
		return self.phasescreen(eta) * fld


class FDPE(object):
	'''
	A class to advance a wavefront through an arbitrary medium using a
	Crank-Nicolson finite difference scheme to solve the parabolic wave
	equation with a quadratically increasing PML along each edge.
	'''
	def __init__(self, k0, nx, ny, h, sigma, l, dz = None):
		'''
		Initialize a class to solve the Crank-Nicolson finite
		difference equations corresponding to the parabolic wave
		equation. The phase variation exp(ikz) is canceled to ensure
		the field is slowly varying along the propagation axis.
		'''
		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0 = h, k0
		# Set the step length
		self.dz = dz if dz else h

		# Compute the PML profile of width l for grid indices c
		def sigprof(c):
			n = c.size
			# The left and right PML boundaries
			o, p = l - 0.5, n - l - 0.5
			# The values c - o, p - c are less than zero inside the PML
			prof = np.minimum(np.minimum(c - o, p - c), 0.)**2
			return 1. + 1j * sigma * h**2 * prof

		# Create some convenient coordinate indices
		x, y = np.ogrid[:nx, :ny]

		# Compute the x and y profiles for the PML
		sp = [[sigprof(c + s * 0.5) for s in [0., 1., -1.]] for c in [x, y]]

		# Replace the PML profiles with the reciprocals of the
		# products required in the second-order differences
		# The first entry of each sigma takes the form sigma[i] * sigma[i + 1/2]
		# The second takes the form sigma[i] * sigma[i - 1/2]
		self.sigx, self.sigy = [[1. / (s[0] * sv) for sv in s[1:]] for s in sp]

		# Compute the scale parameters a- and a+
		self.a = [0.25 * (1. + d * self.k0 * self.dz) for d in [-1j, +1j]]

		# Keep track of the number of slabs
		self.slab = 0


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.ogrid[:g[0], :g[1]]
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	def applydiffs(self, q, fld, left = True):
		'''
		Apply the finite-difference operator to a field fld propagating
		through a medium with acoustic contrast q.

		If left is True, the operator is applied at the next slab using
		the a- parameter (1 - ikz); otherwise, the operator is applied
		at the current slab with the a+ operator (1 + ikz).
		'''
		a = self.a[0] if left else self.a[1]
		ap = a / (self.k0 * self.h)**2

		# Compute all of the self, unstretched terms first
		res = (1. + a * q) * fld

		# Add the central part of the derivatives
		res -= ap * (self.sigx[0] + self.sigx[1]) * fld
		res -= ap * (self.sigy[0] + self.sigy[1]) * fld

		# Add the side terms of the derivatives
		res[1:-1,:] += ap * self.sigx[0][1:-1,:] * fld[2:,:]
		res[1:-1,:] += ap * self.sigx[1][1:-1,:] * fld[:-2,:]
		res[:,1:-1] += ap * self.sigy[0][:,1:-1] * fld[:,2:]
		res[:,1:-1] += ap * self.sigy[1][:,1:-1] * fld[:,:-2]

		# Enforce Dirichlet boundary conditions
		res[0,:] = 0
		res[-1,:] = 0
		res[:,0] = 0
		res[:,-1] = 0

		return res


	def matvec(self, vec):
		'''
		Apply the finite-difference operator to a field propagating
		through the next slab in a Crank-Nicolson scheme. The vector
		vec should hold the field, flattened in FORTRAN order, and any
		index of refraction should have been assigned to the "eta"
		attribute of the class instance. Otherwise, the index of
		refraction is assumed to be unity.
		'''
		vv = np.reshape(vec, self.grid, order='F')
		try: q = self.q
		except AttributeError: q = np.ones_like(vv)
		vv = self.applydiffs(q, vv)
		return np.ravel(vv, order='F')


	def advance(self, fld, q):
		'''
		Advance the field fld through a slab with index of refraction
		eta by solving the sparse linear system resulting from
		Crank-Nicolson finite differences of the parabolic wave
		equation.
		'''
		self.q = q
		# Advance the current slice for accurate phase calculation
		self.slab += 1
		# Compute the differences of the current field
		rhs = self.applydiffs(q, fld, False)
		# Create a LinearOperator to represent the matrix equation
		n = self.grid[0] * self.grid[1]
		op = la.LinearOperator((n,n), matvec=self.matvec, dtype=rhs.dtype)
		nfld, info = la.gmres(op, rhs.ravel('F'),
				tol = 1e-3, maxiter = 50, x0 = fld.ravel('F'))
		return np.reshape(nfld, self.grid, order='F')


	def getphase(self):
		'''
		Return the phase exp(ikz) corresponding to the current slab.
		'''
		return np.exp(1j * self.k0 * self.slab * self.dz)
