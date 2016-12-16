import cython
cimport cython

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

from itertools import izip, product as iproduct

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double minnbr(double[:] t, unsigned long i):
	'''
	For a 1-D array t of length n, return

		min(t[i - 1], t[i + 1])

	if both are legitimate indices into t. If only one is a valid index,
	the corresponding value is returned. If neither is valid, a ValueError
	will be raised.
	'''
	cdef long l, r

	l, r = i - 1, i + 1

	if r >= t.shape[0]:
		if l < 0: raise ValueError('No in-bounds neighbor exists')
		return t[l]

	if l < 0: return t[r]

	cdef double lt = t[l]
	cdef double rt = t[r]
	if lt < rt: return lt
	else: return rt

class FastSweep(object):
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
	@cython.cdivision(True)
	@cython.embedsignature(True)
	def sweep(self, rt not None, rs not None,
			unsigned int octant, inplace=False):
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

		when 1 <= octant <= 8. The directions of axial sweeps increase
		(decrease) when their respective bit values are 0 (1).

		If inplace is True, the solution will be updated in place if
		the input rt is already a suitable array (i.e., the output of
		asarray(rt, dytpe=float64) is either rt or a view on rt). In
		this case, the first return value of this method will be
		identical to rt. The value of inplace is ignored if asarray
		creates a new array.
		'''
		cdef unsigned long nx, ny, nz, updates
		cdef long i, j, k, di, dj, dk
		cdef double hm, fh, Aq, Bq, Cq, nt, a, b, c, hx, hy, hz

		if not 1 <= octant <= 8:
			raise ValueError('Argument "octant" must satisfy 1 <= octant <= 8')

		# Pull and check the grid sizes
		nx, ny, nz = self.box.ncell
		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Grid length must be at least 2 along all axes')

		# Convert the arrays and check types
		cdef double[:,:,:] t
		if inplace: t = np.asarray(rt, dtype=np.float64)
		else: t = np.array(rt, copy=True, dtype=np.float64)

		if t.shape[0] != nx or t.shape[1] != ny or t.shape[2] != nz:
			raise ValueError('Shape of "t" must match grid')

		cdef double[:,:,:] s = np.asarray(rs, dtype=np.float64)

		if s.shape[0] != nx or s.shape[1] != ny or s.shape[2] != nz:
			raise ValueError('Shape of "s" must match grid')

		# Find minimum step size and scale others
		hx, hy, hz = self.box.cell
		hm = hx
		if hy < hm: hm = hy
		if hz < hm: hm = hz
		hx, hy, hz = hx / hm, hy / hm, hz / hm

		octant -= 1

		# Determine the direction of sweeps
		di = 1 if not(octant & 0x1) else -1
		dj = 1 if not(octant & 0x2) else -1
		dk = 1 if not(octant & 0x4) else -1

		# Accumulate a count of updated grid points
		updates = 0

		# k-slices of t are independent of the inner loop
		# i- and j-slices depend on inner loop and cannot be cached
		cdef double[:] tk

		i = 0 if di > 0 else (nx - 1)
		while 0 <= i < nx:
			j = 0 if dj > 0 else (ny - 1)
			while 0 <= j < ny:
				# Pull the relevant k-slice of t
				tk = t[i,j,:]

				k = 0 if dk > 0 else (nz - 1)
				while 0 <= k < nz:
					# Find the neighbors
					a = minnbr(t[:,j,k], i)
					b = minnbr(t[i,:,k], j)
					c = minnbr(tk, k)

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

		return np.asarray(t), updates

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

		s = np.asarray(s, dtype=np.float64)
		if s.shape != ncell:
			raise TypeError('Array "s" must have  shape %s' % (ncell,))

		# Convert source to grid coords and find (extended) enclosing cell
		src = box.cart2cell(*src)
		sgrd = tuple(max(0, min(int(c), n - 1)) for c, n in izip(src, ncell))

		slw = s[sgrd]
		h = box.cell

		# Initialize arrival times to beyond-possible values
		t = np.empty(s.shape, dtype=np.float64, order='C')
		t[:,:,:] = float('inf')

		# Assign arrival-time values to grid points surrounding cell
		for inc in iproduct(xrange(2), repeat=3):
			# Check that the grid point is valid
			npt = tuple(c + i for c, i in izip(sgrd, inc))
			if any(x >= n for x, n in izip(npt, ncell)): continue
			t[npt] = slw * sqrt(sum((hv * (c - i))**2
						for c, i, hv in izip(src, npt, h)))

		it = 0
		while True:
			updates = 0
			for octant in xrange(1, 9):
				t, upc = self.sweep(t, s, octant, True)
				if report:
					print 'Iteration %d (%d): %d updates' % (it, octant, upc)
				updates += upc
			it += 1
			if not updates: break

		return t
