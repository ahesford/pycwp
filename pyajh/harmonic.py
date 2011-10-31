'''
General-purpose numerical routines, relating to angular functions defined on
surfaces of spheres, used in other parts of the module.
'''

import math, numpy as np
from scipy import special as spec
from . import cutil


class SphericalInterpolator:
	'''
	Build a sparse matrix that can be used to interpolate an angular
	function defined on the surface of a sphere.
	'''

	def __init__(self, thetas, order=4, poles=True):
		'''
		Build the Lagrange interpolation matrix of a specified order
		for a regularly sampled angular function. Interpolation windows
		wrap around the pole.

		The 2-element list of lists thetas specifies the locations of
		polar samples for the coarse (thetas[0]) and fine (thetas[1])
		grids. The azimuthal samples at each of the levels are have
		2 * (len(thetas[i]) - 2) regularly spaced values (when poles is
		True) or 2 * len(thetas[i]) values (when poles is False) for
		the corresponding polar samples thetas[i].

		If poles is true, thetas[i][0] and thetas[i][-1] correspond to
		polar values that will only be sampled once. Otherwise,
		thetas[i][0] and thetas[i][-1] have values away from the poles
		and are not treated specially.

		The sparse output matrix format is a list of lists of two
		lists. For a sparse matrix list mat, the list r = mat[i]
		corresponds to the i-th row of the matrix. List r[0] specifies
		the column index of all non-empty entries in the row. List r[1]
		specifies the corresponding values (Lagrange interpolation
		weights) at the nonzero columns.
		'''

		if order > len(thetas[0]):
			raise ValueError('Order should not exceed number of coarse samples.')

		# Grab the total number of polar samples
		ntheta = [len(t) for t in thetas]
		# Grab the azimuthal samples and total samples depending on
		# whether polar values are included
		if poles:
			nphi = [2 * (n - 2) for n in ntheta]
			nsamp = [2 + (nt - 2) * np for nt, np in zip(ntheta, nphi)]
		else:
			nphi = [2 * n for n in ntheta]
			nsamp = [nt * np for nt, np in zip(ntheta, nphi)]

		# Grab the azimuthal step size
		dphi = [2 * math.pi / n for n in nphi]

		# Half the Lagrange interval width
		offset = (order - 1) / 2

		# Simply copy the north pole into the higher sampling rate
		self.matrix = [[[0], [1]]] if poles else []

		# Loop through all polar samples away from the poles
		for rtheta in (thetas[1][1:-1] if poles else thetas[1]):
			# Find the starting interpolation interval
			tbase = cutil.rlocate(thetas[0], rtheta) - offset
			# Enumerate all polar indices involved in interpolation
			rows = [tbase + l for l in range(order)]

			# Build the corresponding angular positions
			tharr = [unwraptheta(thetas[0], ti) for ti in rows]

			# Build the Lagrange interpolation coefficients
			twts = cutil.lagrange(rtheta, tharr)

			# Loop over the higher azimuthal sampling rate
			for j in range(nphi[1]):
				# Initialize the empty matrix row
				matrow = [[], []]
				# Take care of each theta sample
				for tw, rv in zip(twts, rows):
					# Pole samples are not azimuthally interpolated
					if rv == 0 and poles:
						matrow[0].append(0)
						matrow[1].append(tw)
						continue
					elif rv == ntheta[0] - 1 and poles:
						matrow[0].append(nsamp[0] - 1)
						matrow[1].append(tw)
						continue

					# Compute the angular position
					rphi = j * dphi[1]
					# Find the starting interpolation interval
					k = (j * nphi[0]) / nphi[1] - offset

					if rv < 0 or rv >= ntheta[0]:
						rphi += math.pi
						k += nphi[0] / 2

					ri = polarwrapidx(ntheta[0], rv)

					# Build the wrapped phi indices
					cols = [(k + m + nphi[0]) % nphi[0]
							for m in range(order)]
					# Build the unwrapped phi values
					pharr = [(k + m) * dphi[0] for m in range(order)]

					# Build the Lagrange interpolation coefficients
					pwts = cutil.lagrange(rphi, pharr)

					# Populate the columns of the sparse array
					for pw, cv in zip(pwts, cols):
						vpos = linidx(ntheta[0], nphi[0], ri, cv, poles)
						matrow[0].append(vpos)
						matrow[1].append(pw * tw)

				# Add the populated row to the matrix
				self.matrix.append(matrow)

		# Add the last pole value
		if poles: self.matrix.append([[nsamp[0] - 1], [1]])


	def applymat(self, f):
		'''
		Interpolate the coarsely sampled angular function f.
		'''
		# Duplicate the input as a NumPy array, if necessary
		if not isinstance(f, np.ndarray): f = np.array(f)
		# Compute each row of the output
		return np.array([np.dot(r[1], f[r[0]]) for r in self.matrix])


def polarwrapidx(nt, ti, poles=True):
	'''
	For a list of polar samples with length nt, wrap the index ti (possibly
	out of bounds) around the poles to point to an inbounds sample. If
	poles is true, the first and last elements correspond to the poles.
	Otherwise, the poles are not included in the sampling.

	The value may only wrap around a single pole.
	'''

	# Remember to shift negative samples by one if a pole isn't present
	if ti < 0: return -ti if poles else -(ti + 1)
	# Shift one extra sample to account for end-of-list poles
	if ti >= nt: return nt - ti - (2 if poles else 1)

	# By default, no shifting is necessary
	return ti


def unwraptheta(th, ti, poles=True):
	'''
	Given an unwrapped (possibly out-of-bounds) index ti into an array th
	of polar samples, return a corresponding unwrapped value that extends
	beyond the poles. If poles is true, the array contains polar values at
	the first and last indices of th. Otherwise, the poles are not stored.
	'''
	# Grab the length and the wrapped index
	n = len(th)
	tr = polarwrapidx(n, ti, poles)

	# Handle the polar shifts based on the direction of the array
	if th[0] < th[-1]: hs, ls = 2 * math.pi, 0.
	else: hs, ls = 0., 2 * math.pi

	if 0 <= ti < n: return th[ti]
	elif ti < 0: return ls - th[tr]
	else: return hs - th[tr]


def polararray(ntheta):
	'''
	Return a list of polar angular samples corresponding to Gauss-Lobatto
	quadrature points (including poles) in decreasing order.
	'''
	return list(reversed(cutil.gausslob(ntheta)[0]))


def linidx(ntheta, nphi, ti, pi, poles=True):
	'''
	Compute the linearized index the spherical indices (ti, pi), with phi
	most rapidly varying. If poles is True, poles are included, with the
	first pole at index 0 and the second pole at Python index -1. Each pole
	has a single value for ti = 0 or ti = ntheta - 1.

	The index will not be wrapped. Python negative indexing is supported.
	'''

	if not (-ntheta <= ti < ntheta): raise IndexError('Polar index out of bounds.')
	if not (-nphi <= pi < nphi): raise IndexError('Azimuthal index out of bounds.')

	# Fix negative indices according to Python rules
	if ti < 0: ti = ntheta + ti
	if pi < 0: pi = nphi + pi

	# Without a pole, the 0 and -1 values do not degenerate
	if not poles: return ti * nphi + pi
	# With a pole, adjust for the single value at the poles
	else: return 1 + (ti - 1) * nphi + pi


def legassoc (n, m, th):
	'''
	Compute the normalized associated Legendre functions up to degree
	n and order n for an argument cos(th), where th is a polar angle.
	Output is an array with shape (m+1,n+1).
	'''
	t = math.cos(th)
	u = -math.sin(th)

	# Initialize the output array
	lgp = np.zeros((m+1,n+1))

	# The initial value
	lgp[0,0] = math.sqrt(1. / 4. / math.pi)

	# Set up the diagonal elements
	for l in xrange(1,min(m,n)+1):
		lgp[l,l] = math.sqrt((2. * l + 1.) / (2. * l)) * u * lgp[l-1,l-1]

	# Set up the upper diagonal
	for l in xrange(0,min(m,n)):
		lgp[l,l+1] = math.sqrt(2. * l + 3.) * t * lgp[l,l]

	# Now fill in the rest of the matrix
	for p in xrange(0,m+1):
		for l in xrange(1+p,n):
			# Precompute the recursion coefficients
			an = math.sqrt((2. * l + 1.) * (2. * l + 3) /
					(1. + l + p) / (1. + l - p))
			bn = math.sqrt((2. * l + 3.) * (l - p) * (l + p) /
					(2. * l - 1.) / (1. + l + p) / (1. + l - p))
			lgp[p,l+1] = an * t * lgp[p,l] - bn * lgp[p,l-1]

	return lgp

def harmorder (maxdeg):
	'''
	A coroutine that enumerates the orders of a harmonic of a specified
	degree in the order expected by Fastsphere.
	'''
	curdeg = 0

	# Count out the positive orders
	while curdeg <= maxdeg:
		yield curdeg
		curdeg += 1

	# Initialize the negative orders
	curdeg = -maxdeg

	# Count out the negative orders
	while curdeg < 0:
		yield curdeg
		curdeg += 1

	# No more counting to do
	raise StopIteration

def sh2fld (k, clm, r, t, p, reg = True):
	'''
	Expand spherical harmonic coefficients clm for a wave number k over
	the grid range specified by spherical coordinates (r,t,p). Each
	coordinate should be a single-dimension array. If reg is False, use
	a singular expansion. Otherwise, use a regular one.
	'''

	# Pull out the maximum degree and the required matrix leading dimension
	deg, lda = clm.shape[1], 2 * clm.shape[1] - 1

	# If there are not enough harmonic orders, raise an exception
	if clm.shape[0] < lda:
		raise IndexError('Not enough harmonic coefficients.')

	# Otherwise, compress the coefficient matrix to eliminate excess values
	if clm.shape[0] > lda:
		clm = np.array([[clm[i,j] for j in xrange(deg)]
			for i in harmorder(deg-1)])

	# Compute the radial term
	if reg:
		# Perform a regular expansion
		jlr = np.array([spec.sph_jn(deg-1, k*rx)[0] for rx in r])
	else:
		# Perform a singular expansion
		jlr = np.array([spec.sph_jn(deg-1, k*rx)[0] +
			1j * spec.sph_yn(deg-1, k*rx)[0] for rx in r])

	# Compute the azimuthal term
	epm = np.array([[np.exp(1j * m * px) for px in p] for m in harmorder(deg-1)])

	shxp = lambda c, y: np.array([[c[m,l] * y[abs(m),l]
		for l in xrange(deg)] for m in harmorder(deg-1)])

	# Compute the polar term and multiply by harmonic coefficients
	ytlm = np.array([shxp(clm,legassoc(deg-1,deg-1,tx)) for tx in t])

	# Return the product on the specified grid
	fld = np.tensordot(jlr, np.tensordot(ytlm, epm, axes=(1,0)), axes=(1,1))
	return fld.squeeze()
