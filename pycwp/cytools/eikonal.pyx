import cython
cimport cython

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

from itertools import izip, product as iproduct
from numpy import linalg

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.float64_t minnbr(np.ndarray[np.float64_t, ndim=3] t, unsigned int axis,
		unsigned long i, unsigned long j, unsigned long k):
	'''
	Find the neighbor of t[i,j,k], along the specified axis, with minimum
	value. When (i, j, k) is along a boundary, the neighbor away from the
	boundary is always returned.

	No safety check is performed to verify that at least one neighbor
	exists (i.e., that the thickness of the array is at least 2 and that
	(i, j, k) is a valid entry in the array.
	'''
	cdef unsigned long nx, ny, nz
	cdef long ri, rj, rk, li, lj, lk
	nx, ny, nz = t.shape[0], t.shape[1], t.shape[2]

	ri, rj, rk = i, j, k
	li, lj, lk = i, j, k

	if axis == 0:
		ri, li = i + 1, i - 1
	elif axis == 1:
		rj, lj = j + 1, j - 1
	elif axis == 2:
		rk, lk = k + 1, k - 1
	else: raise ValueError('Value of "axis" must satisfy 0 <= axis <= 2')

	if ri >= nx or rj >= ny or rk >= nz:
		return t[li, lj, lk]
	if li < 0 or lj < 0 or lk < 0:
		return t[ri, rj, rk]

	cdef double lt = t[li, lj, lk]
	cdef double rt = t[ri, rj, rk]
	if lt < rt: return lt
	else: return rt

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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def sweep(self, rt not None, rs not None, unsigned int octant, inplace=False):
		'''
		Perform one fast sweep, with axial directions determined by the
		quadrant argument (which must satisfy 1 <= octant < 8), to
		update the Eikonal rt for a slowness (inverse speed) rs.

		The return value is the tuple (nt, count), where nt is the
		updated Eikonal and count is the number of modified grid
		values; if count is 0, no values were changed in the sweep.

		Both rt and rs must be 3-D Numpy arrays with a shape that
		matches self.box.ncell, or compatible sequences that will be
		converted as necessary.

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
		the input rt is already a suitable array (i.e., the output of
		asarray(rt) is either rt or a view on rt). In this case, the
		first return value of this method will be identical to rt. The
		value of inplace is ignored if asarray(rt) creates a new array.
		'''
		cdef unsigned long nx, ny, nz, updates
		cdef long i, j, k, di, dj, dk
		cdef double hm, fh, Aq, Bq, Cq, nt, a, b, c, hx, hy, hz

		if not 1 <= octant <= 8:
			raise ValueError('Argument "octant" must satisfy 1 <= octant <= 8')

		# Pull and check the grid sizes
		grid = self.box.ncell
		nx, ny, nz = grid[0], grid[1], grid[2]
		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Grid length must be at least 2 along all axes')

		# Convert the arrays and check types
		cdef np.ndarray[np.float64_t, ndim=3] t
		if inplace: t = np.asarray(rt, dtype=np.float64)
		else: t = np.array(rt, dtype=np.float64)

		if t.shape[0] != nx or t.shape[1] != ny or t.shape[2] != nz:
			raise ValueError('Shape of "t" must match grid')

		cdef np.ndarray[np.float64_t, ndim=3] s = np.asarray(rs, dtype=np.float64)

		if s.shape[0] != nx or s.shape[1] != ny or s.shape[2] != nz:
			raise ValueError('Shape of "s" must match grid')

		# Find minimum step size and scale others
		cell = self.box.cell
		hx, hy, hz = cell[0], cell[1], cell[2]
		hm = hx
		if hy < hm: hm = hy
		if hz < hm: hm = hz
		hx, hy, hz = hx / hm, hy / hm, hz / hm

		octant -= 1

		# Determine the direction of sweeps
		di = 1 if not(octant & 0x1) else -1
		dj = 1 if not(octant & 0x2) else -1
		dk = 1 if not(octant & 0x4) else -1

		updates = 0
		i = 0 if di > 0 else (nx - 1)
		while 0 <= i < nx:
			j = 0 if dj > 0 else (ny - 1)
			while 0 <= j < ny:
				k = 0 if dk > 0 else (nz - 1)
				while 0 <= k < nz:
					# Find the neighbors
					a = minnbr(t, 0, i, j, k)
					b = minnbr(t, 1, i, j, k)
					c = minnbr(t, 2, i, j, k)

					ha, hb, hc = hx, hy, hz

					if a > b:
						# Ensure a <= b
						a, b = b, a
						ha, hb = hb, ha

					if b > c:
						# Ensure b <= c
						b, c = c, b
						hb, hc = hc, hb

						if a > b:
							# Recheck a <= b
							a, b = b, a
							ha, hb = hb, ha

					# First check single-axis solution
					fh = s[i,j,k] * hm
					nt = a + fh * ha

					# If solution too large, try two axes
					if nt > b:
						# RHS and grid are only used squared
						fh *= fh
						ha *= ha
						hb *= hb

						# Find terms in quadratic formula
						Aq = ha + hb
						Bq = hb * a + ha * b
						Cq = hb * a * a + ha * (b * b - fh * hb)

						nt = (Bq + sqrt(Bq * Bq - Aq * Cq)) / Aq

						# If still too large, need all three axes
						if nt > c:
							# Final axis scale still needs squaring
							hc *= hc
							# Find terms in quadratic formula
							Aq = hb * hc + ha * hc + ha * hb
							Bq = hb * hc * a + ha * (hc * b + hb * c)
							Cq = (hb * hc * (a * a - fh * ha) +
								ha * (hc * b * b + hb * c * c))
							nt = (Bq + sqrt(Bq * Bq - Aq * Cq)) / Aq

					if nt < t[i,j,k]:
						updates += 1
						t[i,j,k] = nt
					k += dk
				j += dj
			i += di

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

		# Initialize arrival times to beyond-possible values
		t = np.empty_like(s)
		t[:,:,:] = float('inf')

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
