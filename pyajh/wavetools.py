'''
Routines useful for wave physics computations, including CG-FFT, phase and 
magnitude correction for comparison of k-space (FDTD) and integral equation
solvers, and Green's functions.
'''

import math, cmath, numpy as np, scipy as sp
import numpy.fft as fft, scipy.special as spec, scipy.sparse.linalg as la
from numpy.linalg import norm

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
	def duffcell(s, d):
		l = 0.5 * d[0] - s[0]
		dc = [l] + [dv / l for dv in d[1:]]
		src = [0.5 * l] + [-sv / l for sv in s[1:]]
		return src, dc

	# Store the final integration value
	val = 0.

	# Deal with each axis in succession
	for i in range(dim):
		# Integrate over the pyramid along the +x axis
		src, dc = duffcell(obs, cell)
		# The obs argument is ignored so it is set to 0
		val += srcint(k, src, [0.]*dim, dc, grf, n)

		# Integrate over the pyramid along the -x axis
		src, dc = duffcell([-obs[0]] + obs[1:], cell)
		# The obs argument is ignored so it is set to 0
		val += srcint(k, src, [0.]*dim, dc, grf, n)

		# Rotate the next axis into the x position
		obs = list(obs[1:]) + list(obs[:1])
		cell = list(cell[1:]) + list(cell[:1])

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

def extgreen(k, grid, cell, greenfunc = green3d):
	'''
	Compute the extended Green's function with wave number k for use in
	CG-FFT over a non-extended grid with dimensions specified in the list
	grid. Each cell has dimensions specified in the list cell.

	The Green's function greenfunc(k,r) takes as arguments the wave number
	and a scalar distance or numpy array of distances between the source
	and observation locations. The 3-D Green's function is used by default.
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
	grf = greenfunc(k, r) * np.prod(cell)

	# Define a pairwise Green's function for integration
	def greenpair(kv, x, y):
		return greenfunc(kv, norm([xl - yl for xl, yl in zip(x,y)]))

	# Correct the zero value to remove the singularity
	grf[[slice(1) for d in range(dim)]] = duffyint(k, [0.]*dim, cell)

	# Return the FFT of the extended-grid Green's function
	return fft.fftn(k**2 * grf)

def applygrf(fld, grf):
	'''
	Apply the Green's function grf to the field fld using FFT convolution.
	'''
	if len(fld.shape) not in (2, 3):
		raise ValueError('Problem must be 2-D or 3-D.')
	if len(fld.shape) != len(grf.shape):
		raise ValueError('Arguments must have same dimensionality.')

	# Compute the convolution with the Green's function
	efld = fft.ifftn(grf * fft.fftn(fld, s = grf.shape))

	# Return the relevant portion of the solution
	if len(fld.shape) == 2: return efld[:fld.shape[0], :fld.shape[1]]
	return efld[:fld.shape[0], :fld.shape[1], :fld.shape[2]]

def scatmvp(fld, grf, obj):
	'''
	Apply the scattering operator to an input field using FFT convolution.
	'''
	return fld - applygrf(fld * obj, grf)

def solve(itfunc, grf, obj, rhs, **kwargs):
	'''
	Solve the scattering problem for a precomputed Green's function grf,
	an object contrast (potential) obj, and an incident field rhs. The
	iterative solver itfunc is used, from scipy.sparse.linalg.
	'''

	# Compute the dimensions of the linear system
	n = np.prod(obj.shape)

	# Function to compute the matrix-vector product
	mvp = lambda v: scatmvp(v.reshape(obj.shape, order='F'), grf, obj).flatten('F')

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
