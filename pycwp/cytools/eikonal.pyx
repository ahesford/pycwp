# Copyright (c) 2016--2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import cython
cimport cython

from cython cimport floating as real

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

from itertools import product as iproduct

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef real minnbr(real[:] t, unsigned long i) nogil:
	'''
	For a 1-D array t of length n, return

		min(t[i - 1], t[i + 1])

	if both are legitimate indices into t. If only one is a valid index,
	the corresponding value is returned. If neither is valid, infinity will
	be returned.
	'''
	cdef long l, r

	l, r = i - 1, i + 1

	if r >= t.shape[0]:
		if l < 0: return 1.0 / 0.0
		return t[l]

	if l < 0: return t[r]

	cdef real lt = t[l]
	cdef real rt = t[r]
	if lt < rt: return lt
	else: return rt

class FastSweep(object):
	'''
	A class to represent a 3-D solution to the Eikonal equation on a
	regular grid (as a pycwp.boxer.Box3D object) using the fast sweeping
	method in Zhao, "A fast sweeping method for Eikonal equations", Math.
	Comp. 74 (2005), 603--627.
	'''
	@cython.embedsignature(True)
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
	def sweep(self, real[:,:,:] t not None, real[:,:,:] s not None,
			unsigned int octant, bint inplace=False):
		'''
		FastSweep.sweep(t, s, octant, inplace=False)

		Perform one fast sweep, with axial directions determined by the
		quadrant argument (which must satisfy 0 <= octant <= 7), to
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
		bitfield into (x, y, z) bit values using the relationship

			octant = (z << 2) & (y << 1) & x.

		The directions of axial sweeps increase (decrease) when their
		respective bit values are 0 (1).

		If inplace is True, the solution will be updated in place if t
		is already a suitable floating-point array. In this case, the
		returned nt will be identical to t. If inplace is False, a copy
		of t will be made, updated, and returned.
		'''
		cdef unsigned long nx, ny, nz, updates
		cdef long i, j, k, di, dj, dk
		cdef real hm, fh, Aq, Bq, Cq, nt, a, b, c, hx, hy, hz

		if not octant < 8:
			raise ValueError('Argument "octant" must satisfy 0 <= octant <= 7')

		# Pull and check the grid sizes
		nx, ny, nz = self.box.ncell
		if nx < 2 or ny < 2 or nz < 2:
			raise ValueError('Grid length must be at least 2 along all axes')

		if t.shape[0] != nx or t.shape[1] != ny or t.shape[2] != nz:
			raise ValueError('Shape of "t" must match grid')

		if s.shape[0] != nx or s.shape[1] != ny or s.shape[2] != nz:
			raise ValueError('Shape of "s" must match grid')

		# Make a copy of the input Eikonal if desired
		if not inplace: t = t.copy()

		# Find minimum step size and scale others
		hx, hy, hz = self.box.cell
		hm = hx
		if hy < hm: hm = hy
		if hz < hm: hm = hz
		hx, hy, hz = hx / hm, hy / hm, hz / hm

		# Determine the direction of sweeps
		di = 1 if not(octant & 0x1) else -1
		dj = 1 if not(octant & 0x2) else -1
		dk = 1 if not(octant & 0x4) else -1

		# Accumulate a count of updated grid points
		updates = 0

		# k-slices of t are independent of the inner loop
		# i- and j-slices depend on inner loop and cannot be cached
		cdef real[:] tk
		cdef real ha, hb, hc

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

	@cython.boundscheck(False)
	@cython.wraparound(False)
	def gauss(self, src, real[:,:,:] s, bint report=False):
		'''
		FastSweep.gauss(self, src, s, report=False)

		Given a source location src in Cartesian coordinates and a
		slowness s, a 3-D real array with shape self.box.ncell, compute
		and return the Eikonal t (a Numpy array with a shape matching
		that of s) that describes arrival times from the source to
		every point in the domain according to the slowness map.
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

		cdef unsigned long nx, ny, nz
		nx, ny, nz = ncell

		if s.shape[0] != nx or s.shape[1] != ny or s.shape[2] != nz:
			raise TypeError('Array "s" must have shape %s' % (ncell,))

		# Initialize arrival times to beyond-possible values
		if real is float: dtype = np.dtype('float32')
		elif real is double: dtype = np.dtype('float64')
		else: raise TypeError('Type of "s" must be float or double')

		t = np.empty(ncell, dtype=dtype, order='C')
		t[:,:,:] = float('inf')

		cdef real sx, sy, sz
		cdef real mx, my, mz
		sx, sy, sz = src
		mx, my, mz = box.midpoint

		# Start sweeping in the dominant direction
		cdef unsigned long soct
		soct = <unsigned int>(mz < sz)
		soct <<= 1
		soct |= <unsigned int>(my < sy)
		soct <<= 1
		soct |= <unsigned int>(mx < sx)

		# Convert source to grid coords and find (extended) enclosing cell
		sx, sy, sz = box.cart2cell(sx, sy, sz)

		cdef unsigned long si, sj, sk
		si = min(<unsigned long>max(<long>sx, 0), nx - 1)
		sj = min(<unsigned long>max(<long>sy, 0), ny - 1)
		sk = min(<unsigned long>max(<long>sz, 0), nz - 1)

		cdef real slw = s[si,sj,sk]
		cdef real hx, hy, hz
		hx, hy, hz = box.cell

		# Assign arrival-time values to grid points surrounding cell
		cdef long i, j, k, ci, cj, ck
		cdef real rx, ry, rz
		for i in range(2):
			ci = si + i
			if ci >= nx: continue
			rx = hx * (sx - <real>ci)
			rx *= rx
			for j in range(2):
				cj = sj + j
				if cj >= ny: continue
				ry = hy * (sy - <real>cj)
				ry *= ry
				for k in range(2):
					ck = sk + k
					if ck >= nz: continue
					rz = hz * (sz - <real>ck)
					rz *= rz
					t[ci,cj,ck] = slw * sqrt(rx + ry + rz)

		cdef unsigned long octant

		it = 0
		while True:
			updates = 0
			for octant in range(soct, soct + 8):
				octant = octant % 8
				t, upc = self.sweep(t, s, octant, True)
				if report:
					print 'Iteration %d (%d): %d updates' % (it, octant, upc)
				updates += upc
			it += 1
			if not updates: break

		return t
