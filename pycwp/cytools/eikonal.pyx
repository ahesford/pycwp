import cython
cimport cython

import numpy as np
cimport numpy as np

from math import sqrt
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef np.float64_t minnbr(np.ndarray[np.float64_t, ndim=3] t, unsigned int axis,
		unsigned long i, unsigned long j, unsigned long k):
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

@cython.boundscheck(False)
@cython.wraparound(False)
def sweep(rt not None, rs not None,
		double hx, double hy, double hz, unsigned int octant):
	'''
	Perform an Eikonal sweep.
	'''
	cdef unsigned long nx, ny, nz, updates
	cdef long i, j, k, di, dj, dk
	cdef double hm, fh, Aq, Bq, Cq, nt, a, b, c

	if not 1 <= octant <= 8:
		raise ValueError('Argument "octant" must satisfy 1 <= octant <= 8')

	# Convert the arrays if possible
	cdef np.ndarray[np.float64_t, ndim=3] t = np.asarray(rt, dtype=np.float64)
	cdef np.ndarray[np.float64_t, ndim=3] s = np.asarray(rs, dtype=np.float64)

	# Pull the grid sizes
	nx, ny, nz = t.shape[0], t.shape[1], t.shape[2]

	if nx < 2 or ny < 2 or nz < 2:
		raise ValueError('Length of all axes of "t" must exceed 1')

	if s.shape[0] != nx or s.shape[1] != ny or s.shape[2] != nz:
		raise ValueError('Arguments "t" and "s" must have same shape')

	# Find minimum step size and scale others
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
