'''
General-purpose numerical routines, relating to angular functions defined on
surfaces of spheres, used in other parts of the module.
'''

import math, numpy as np
from scipy import special as spec


class SphericalInterpolator:
	'''
	Build a sparse matrix that can be used to interpolate an angular
	function defined on the surface of a sphere.
	'''

	def __init__(self, thetas, phis, order=4):
		'''
		Build the Lagrange interpolation matrix of a specified order
		for a regularly sampled angular function. Interpolation windows
		wrap around the pole at most once.

		The lists of lists thetas and phis specify the number of polar
		and azimuthal samples, respectively, of the coarse (index 0) and
		fine (index 1) grids.

		The sparse output matrix format is a list of lists of two
		lists. For a sparse matrix list mat, the list r = mat[i]
		corresponds to the i-th row of the matrix. List r[0] specifies
		the column index of all non-empty entries in the row. List r[1]
		specifies the corresponding values (Lagrange interpolation
		weights) at the nonzero columns.
		'''
		
		if order > len(thetas[0]) or order > len(phis[0]):
			raise ValueError('Order should not exceed number of samples at coarse rate.')


		# Grab the total number of samples
		ntheta = [len(t) for t in thetas]
		nphi = [len(p) for p in phis]
		
		# The number of samples at the lower [0] and higher [1] sampling rates
		nsamp = [2 + (nt - 2) * np for nt, np in zip(ntheta, nphi)]
		
		# Half the Lagrange interval width
		offset = (order - 1) / 2
		
		# Simply copy the north pole into the higher sampling rate
		self.matrix = [[[0], [1]]]
		
		# Loop over the higher polar sampling rate, ignoring the poles
		for rtheta in thetas[1][1:-1]:
			# Find the starting interpolation interval
			tbase = cutil.rlocate(thetas[0], rtheta) - offset
			# Enumerate all polar indices involved in interpolation
			rows = [tbase + l for l in range(order)]
			
			# Build the corresponding angular positions, unwrapped
			tharr = [harmonic.polarwrap(thetas[0], ti) for ti in rows]
			
			# Build the Lagrange interpolation coefficients
			twts = cutil.lagrange(rtheta, tharr)
			
			# Loop over the higher azimuthal sampling rate
			for rphi in phis[1]:
				# Initialize the empty matrix row
				matrow = [[], []]
				# Take care of each theta sample
				for tw, rv in zip(twts, rows):
					# Pole samples are not azimuthally interpolated
					if rv == 0:
						matrow[0].append(0)
						matrow[1].append(tw)
						continue
					elif rv == ntheta[0] - 1:
						matrow[0].append(nsamp[0] - 1)
						matrow[1].append(tw)
						continue

					# Find the starting interpolation interval
					phiv = rphi
					k = cutil.rlocate(phis[0], phiv) - offset
					
					# Adjust for crossing the pole
					if not (0 <= rv < ntheta[0]):
						phiv += math.pi
						k += ntheta[0] / 2

					# Now correct the polar index
					if rv >= ntheta[0]:
						rv = 2 * (ntheta[0] - 1) - rv
					elif rv < 0: rv = abs(rv)
						
					# Build the wrapped phi indices
					cols = [(k + m + nphi[0]) % nphi[0]
							for m in range(order)]
					# Build the unwrapped phi values
					pharr = [harmonic.aziwrap(phis[0], k + m)
							for m in range(order)]
					
					# Build the Lagrange interpolation coefficients
					pwts = cutil.lagrange(phiv, pharr)

					# Populate the columns of the sparse array
					for pw, cv in zip(pwts, cols):
						vpos = harmonic.linidx(ntheta[0], nphi[0], rv, cv)
						matrow[0].append(vpos)
						matrow[1].append(pw * tw)
						
				# Add the populated row to the matrix
				self.matrix.append(matrow)

		# Copy the south pole value
		self.matrix.append([[nsamp[0] - 1], [1]])


	def applymat(self, f):
		'''
		Interpolate the coarsely sampled angular function f.
		'''

		# Loop through all rows to compute the total output
		return [cutil.dot(r[1], (f[i] for i in r[0])) for r in self.matrix]


def linidx(ntheta, nphi, ti, pi, poles=True):
	'''
	Compute the linearized index the spherical indices (ti, pi), with phi
	most rapidly varying. If poles is True, poles are included, with the
	first pole at index 0 and the second pole at index -1.

	The index will not be wrapped.
	'''

	if not (0 <= ti < ntheta): raise IndexError('Polar index out of bounds.')
	if not (0 <= pi < nphi): raise IndexError('Azimuthal index out of bounds.')

	# Without a pole, the 0 and -1 values do not degenerate
	if not poles: return ti * nphi + pi
	# With a pole, adjust for the single value at the poles
	else: return 1 + (ti - 1) * nphi + pi


def polarwrap(thetas, i):
	'''
	This returns the angular value corresponding to index i wrapped around
	a pole. For correct results, a polar value (v = 0 or v = pi) must exist
	at the end of the list around which the index is wrapped.

	If i < 0, a polar value must reside at thetas[0].
	If i >= len(thetas), a polar value must reside at thetas[-1].
	'''
	n = len(thetas)

	# Check the limits
	if i < 1 - n or i > 2 * (n - 1):
		raise ValueError('Index may only wrap around one pole.')

	# Don't wrap if the index is properly placed
	if 0 <= i < n: return thetas[i]

	# For increasing arrays, the north pole (theta = 0) is at the start
	if thetas[0] < thetas[-1]: ls, hs = 0, 2. * math.pi
	# For decreasing arrays, the north pole is at the end
	else: ls, hs = 2. * math.pi, 0.

	# For negative indices, shift around the low value
	if i < 0: return ls - thetas[-i]
	# For too-high indices, shift around the high value
	return hs - thetas[2 * n - i - 2]


def aziwrap(phis, i):
	'''
	This returns the angular value corresponding at index i. If i is out of
	bounds, the angle is unwrapped by 2 pi.
	'''
	n = len(phis)
	return 2. * math.pi * int(i / n) + phis[i % n]


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
