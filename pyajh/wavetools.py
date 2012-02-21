'''
Routines useful for wave physics computations, including CG-FFT, phase and 
magnitude correction for comparison of k-space (FDTD) and integral equation
solvers, Green's functions, and a split-step solver.
'''

import math, cmath, numpy as np, scipy as sp
import numpy.fft as fft, scipy.special as spec, scipy.sparse.linalg as la
from numpy.linalg import norm
from itertools import izip

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

def directivity(pos, src, d, a):
	'''
	Evaluate a directivity pattern cos(theta) exp(-a sin(theta)**2), where
	theta is the angle between a direction d and the distance r = (pos -
	src), where pos is the position of the observer and src is the position
	of the source.
	'''
	# Compute the distance r and its norm
	r = [x - y for x, y in izip(src, pos)]
	rn = np.sqrt(reduce(np.add, [rv**2 for rv in r]))
	# Normalize the directivity
	dn = np.sqrt(reduce(np.add, [dv**2 for dv in d]))
	d = [dv / dn for dv in d]

	# Compute the cosine of the angle
	ctheta = reduce(np.add, [x * y for x, y in izip(r, d)]) / rn
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

	# Compute a coordinate grid for sampling within the cell
	coords = np.mgrid[[slice(n) for i in range(dim)]]
	# Flatten the coordinate grid into an array of tuples
	coords = zip(*[c.flat for c in coords])

	# Compute the cell-relative quadrature points
	qpts = [[o + s * wts[i][0] for i, o, s in zip(c, src, sc)] for c in coords]

	# Compute the corresponding quadrature weights
	qwts = [np.prod([wts[i][1] for i in c]) for c in coords]

	# Sum all contributions to the integral
	ival = np.sum(w * ifunc(k, p, obs) for w, p in zip(qwts, qpts))

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
		coords = np.mgrid[[slice(2 * g) for g in grid]].astype(np.float)
		
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


class SplitStepEngine(object):
	'''
	A class to advance a wavefront through an arbitrary medium according to
	the split-step method.
	'''

	def __init__(self, k0, nx, ny, h):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0.
		'''

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0 = h, k0

		# The default propagator and attenuation screen are None
		self._propagator = None
		self._attenuator = None


	def kxysq(self):
		'''
		Return the square of the transverse wave number, caching the
		result if it has not already been computed.
		'''

		try: return self._kxysq
		except AttributeError:
			# Get the coordinate indices for each slab
			nx, ny = self.grid
			sl = np.mgrid[-nx/2:nx/2, -ny/2:ny/2].astype(float)

			# Compute the transverse wave numbers
			kx = 2. * math.pi * sl[0] / (nx * self.h)
			ky = 2. * math.pi * sl[1] / (ny * self.h)

			self._kxysq = fft.fftshift(kx**2 + ky**2)
			return self._kxysq 


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.mgrid[:g[0], :g[1]].astype(float)
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	@property
	def propagator(self):
		'''
		Return a propagator to advance wabes to the next slab.
		'''
		return self._propagator


	@propagator.setter
	def propagator(self, dz):
		'''
		Generate and cache the slab propagator for a step size dz. If
		dz is None, use a step that matches the transverse step.
		'''
		# Make the grid isotropic if slab thickness isn't specified
		if dz is None: dz = self.h

		# Get the longitudinal wave number
		kz = np.sqrt(complex(self.k0**2) - self.kxysq())

		# Compute the propagator
		self._propagator = np.exp(1j * kz * dz)


	@propagator.deleter
	def propagator(self): del self._propagator


	def propfield(self, fld):
		'''
		Apply the spectral propagator to the provided field.
		'''
		return fft.ifftn(self.propagator * fft.fftn(fld))


	@property
	def attenuator(self):
		'''
		Return a screen to attenuate the field at each slab edge. If
		one was not previously set, a screen that does not attenuate is
		created.
		'''
		return self._attenuator


	@attenuator.setter
	def attenuator(self, aparm):
		'''
		Generate a screen to supresses the field in each slab. The
		tuple aparm takes the form (a, t), where a is the maximum
		attenuation exp(-a), and t is the thickness of the border over
		which the attenuation increases from zero to the maximum.
		'''
		# Unfold the parameter tuple
		atten, nt = aparm
		# Build an index grid centered on the origin
		nx, ny = self.grid
		sl = np.mgrid[:nx,:ny].astype(float)
		sl[0] -= (nx - 1.) / 2.
		sl[1] -= (ny - 1.) / 2.
		# Positive points lie within the absorbing boundaries
		x = (np.abs(sl[0]) - (0.5 * nx - nt)) / float(nt)
		y = (np.abs(sl[1]) - (0.5 * ny - nt)) / float(nt)

		# Compute the attenuation profile
		wx = 1 - np.sin(0.5 * math.pi * (x > 0).choose(0, x))**2
		wy = 1 - np.sin(0.5 * math.pi * (y > 0).choose(0, y))**2
		w = wx * wy

		# This is the imaginary part of the wave number in the boundary
		self._attenuator = 1j * atten * (1. - w) / self.k0


	@attenuator.deleter
	def attenuator(self): del self._attenuator


	def phasescreen(self, eta, dz = None):
		'''
		Generate the phase screen for a slab characterized by scaled
		wave number eta. If the step size dz is not specified, the
		isotropic in-plane step is used. Results are not cached.
		'''

		# Use the default step size if necessary
		if dz is None: dz = self.h

		return np.exp(1j * dz * self.k0 * (eta - 1.))


	def wideangle(self, fld, eta, dz = None):
		'''
		Apply wide-angle corrections to the propagated field fld
		corresponding to a medium with scaled wave number eta.
		'''
		if dz is None: dz = self.h

		# Compute the Laplacian term and the spatial correction
		kbar = self.kxysq() / self.k0**2
		cor = 1j * self.k0 * dz * (eta - 1.) / (2. * eta)
		return fld + cor * fft.ifftn(kbar * fft.fftn(fld))


	def advance(self, fld, obj, w = True, dz = None):
		'''
		Use the spectral propagator, the phase screen for the current
		slab, and the attenuation screen to advance the field through
		one layer of a specified medium with contrast obj. The
		propagator and attenuation screen must have previously been
		established.
		'''

		# Convert the contrast to a scaled wave number
		eta = np.sqrt(obj + 1.)

		# Incorporate an absorbing boundary if one was desired
		if self.attenuator is not None: eta += self.attenuator

		# Propagate the field through one step
		fld = self.propfield(fld)

		# Apply wide-angle corrections, if desired
		if w: fld = self.wideangle(fld, eta, dz)

		# Apply the phase screen for this slab
		fld = self.phasescreen(eta, dz) * fld

		return fld
