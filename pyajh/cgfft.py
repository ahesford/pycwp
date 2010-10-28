'''
Routines to use CG-FFT for scattering from objects in two and three dimensions.
'''

import math
import numpy as np
import scipy as sp
import numpy.fft as fft
import scipy.special as spec
import scipy.sparse.linalg as la
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

def srcint(k, src, obs, cell, n = 4):
	'''
	Evaluate source integration, of order n, of the Green's function.
	'''

	if n % 2 != 0: raise ValueError('Order must be even to avoid singularity.')

	dim = len(cell)
	if dim not in (2, 3): raise ValueError('Cell dimensions must be 2-D or 3-D.')

	src, obs, cell = np.array(src), np.array(obs), np.array(cell)

	# Compute the node scaling factor
	sc = 0.5 * cell

	# Compute the offset between source and observation
	off = src - obs

	# Grab the roots and weights of the Legendre polynomial of order n
	wts = spec.legendre(n).weights
	wts = zip(wts[:,1], wts[:,0])

	# Perform the integration in 2-D or 3-D as appropriate
	if dim == 2:
		ival = np.sum([[wx * wy * green2d(k, norm(off + sc * [px, py]))
			for wx, px in wts] for wy, py in wts])
	else:
		ival = np.sum([[[wx * wy * wz * green3d(k, norm(off + sc * [px, py, pz]))
			for wx, px in wts] for wy, py in wts] for wz, pz in wts])

	return ival * np.prod(cell) / 2.**dim

def extgreen(k, grid, cell):
	'''
	Compute the extended Green's function for use in CG-FFT.
	'''

	dim = min(len(grid),len(cell))
	if dim not in (2, 3): raise ValueError('Problem must be 2-D or 3-D.')

	grid, cell = np.array(grid[:dim]), np.array(cell[:dim])

	if dim == 2:
		# Compute the coordinates on the grid
		slice = np.mgrid[0.:2*grid[0], 0.:2*grid[1]]
		x = cell[0] * (slice[0] < grid[0]).choose(slice[0] - 2 * grid[0], slice[0])
		y = cell[1] * (slice[1] < grid[1]).choose(slice[1] - 2 * grid[1], slice[1])

		# Evaluate the Green's function on the grid
		r = np.sqrt(x**2 + y**2)
		grf = green2d(k, r) * np.prod(cell)
		# Correct the zero value to remove the singularity
		grf[0,0] = srcint(k, (0., 0.), (0., 0.), cell)
	else:
		# Compute the coordinates on the grid
		slice = np.mgrid[0.:2*grid[0], 0.:2*grid[1], 0.:2*grid[2]]
		x = cell[0] * (slice[0] < grid[0]).choose(slice[0] - 2 * grid[0], slice[0])
		y = cell[1] * (slice[1] < grid[1]).choose(slice[1] - 2 * grid[1], slice[1])
		z = cell[2] * (slice[2] < grid[2]).choose(slice[2] - 2 * grid[2], slice[2])

		# Evaluate the Green's function on the grid
		r = np.sqrt(x**2 + y**2 + z**2)
		grf = green3d(k, r) * np.prod(cell)
		# Correct the zero value to remove the singularity
		grf[0,0,0] = srcint(k, (0., 0., 0.), (0., 0., 0.), cell)

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
