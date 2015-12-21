'''
General-purpose numerical routines, relating to angular functions defined on
surfaces of spheres, used in other parts of the module.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import math, numpy as np
from scipy import special as spec, sparse
from itertools import count, izip
from . import cutil, poly, quad


class HarmonicSpline(object):
	'''
	Use cubic basis splines to interpolate a harmonic function defined on
	the surface of a sphere.
	'''

	def __init__(self, thetas, tol = 1e-7):
		'''
		Prepare to interpolate functions defined on a coarse grid, with
		polar samples specified in thetas[0], onto a fine grid whose
		polar samples are specified in thetas[1]. For each of the
		coarse grid and the fine grid, the azimuthal samples occur at
		regular intervals with cardinality

			nphi[i] = 2 * (len(thetas[i]) - 2).

		The first and last samples of each element in the thetas list
		correspond to values at the poles and, therefore, only have one
		associated (but arbitrary) azimuthal angle.

		For efficiency, the coefficients are only computed to within a
		tolerance specified by tol. Set tol < 0 for full precision.
		'''

		# Compute the number of angular samples
		self.nthetas = [len(th) for th in thetas]
		self.nphis = [2 * (nt - 2) for nt in self.nthetas]

		# Compute the number of output samples
		osamp = 2 + (self.nthetas[1] - 2) * self.nphis[1]

		# Split the polar arrays
		thetac, thetaf = thetas

		# Ensure the polar samples are increasing
		if thetac[0] > thetac[-1]:
			self.reverse = True
			thetac = thetac[::-1]
		else: self.reverse = False

		# Compute the azimuthal interval width
		dphi = 2. * math.pi / self.nphis[0]

		# The dimensions of the coefficient grid at the coarse level
		n, m = 2 * (self.nthetas[0] - 1), self.nphis[0] / 2

		# Precompute the pole for the causal and anti-causal filters
		zp = math.sqrt(3) - 2.
		# Also precompute all of the necessary powers of the pole
		self.zpn = zp**np.arange(n + 1)

		# Restrict the precision, if desired
		if tol > 0:
			self.precision = int(math.log(tol) / math.log(abs(zp)))
		else: self.precision = max(n, m)

		# Initialize the list of weights and indices
		weights, idx, rval = [], [], 0

		# Loop through all polar samples
		for thi, rtheta in enumerate(thetaf):
			# Find the interval containing the finer polar sample
			i = cutil.rlocate(thetac, rtheta)
			# Ensure the interval doesn't wrap around the poles
			i = min(max(i, 0), self.nthetas[0] - 2)

			# Grab the fractional distance into the interval
			dtheta = thetac[i + 1] - thetac[i]
			alpha = (rtheta - thetac[i]) / dtheta

			# Compute the cubic b-spline weights
			w = self.weights(alpha)

			# The azimuthal samples degenerate at the poles
			if thi == 0 or thi == self.nthetas[1] - 1: nphi = 1
			else: nphi = self.nphis[1]

			# Loop through the azimuthal samples
			for jf in range(nphi):
				# Find the fractional coarse interval
				pa = float(jf * self.nphis[0]) / self.nphis[1]
				# Ensure that the interval doesn't wrap
				j = min(max(int(pa), 0), self.nphis[0] - 1)

				# Grab the fractional distance into the interval
				beta = pa - j

				# Compute the cubic b-spline weights
				u = self.weights(beta)

				# Handle wrapping in the second hemisphere
				if j >= m:
					thwts = izip(reversed(w), count(-i - 2))
					j -= m
				else: thwts = izip(w, count(i - 1))

				for wv, iv in thwts:
					for uv, jv in izip(u, count(j - 1)):
						# Compute the contributing weight
						weights.append(wv * uv)
						# Compute its (wrapped) index
						iw = iv if (0 <= jv < m) else -iv
						ij = (iw % n) + n * (jv % m)
						idx.append([rval, ij])

				rval += 1

		# Create a CSR matrix representation of the interpolator
		self.matrix = sparse.csr_matrix((weights, zip(*idx)), shape=(osamp, n * m))


	def getcoeff(self, f):
		'''
		Given a function f defined on the unit sphere at the coarse
		sampling rate defined in the constructor, compute and return a
		2-D array of coefficients that expand the function in terms of
		the cubic b-spline basis.

		The coefficients have the polar angle along the rows and the
		azimuthal angle along the colums, with the polar angle in the
		interval [0, 2 * pi] and the azimuthal angle in [0, pi].
		'''

		# Note the dimensions of the input grid
		ntheta, nphi = self.nthetas[0], self.nphis[0]

		# Store the poles
		poles = f[0], f[-1]
		# Reshape the remaining samples, theta along the rows
		f = np.reshape(f[1:-1], (ntheta - 2, nphi), order='C')

		# Ensure samples of the polar angle are increasing
		if self.reverse:
			f = f[::-1, :]
			poles = poles[::-1]

		# Grab the pole and its powers
		zpn = self.zpn
		zp = zpn[1]

		# Create the coefficient grid
		n, m, k = 2 * (ntheta - 1), nphi / 2, ntheta - 1
		c = np.zeros((n, m), dtype=f.dtype)

		# Copy the first hemisphere of data
		c[1:k, :] = f[:, :m]
		# Copy the second hemisphere of data with flipped polar angle
		c[k+1:, :] = f[-1::-1, m:]

		# Copy the poles into the appropriate rows
		c[0, :] = poles[0]
		c[k, :] = poles[-1]

		# Compute the filter coefficients
		l = 6. / (1 - zpn[n]), zp / (zpn[n] - 1)

		# Limit the number of terms in the sum
		p = min(n - 1, self.precision)

		# Compute the initial causal polar coefficient
		# c[0] is never in the dot product since p < n
		c[0] = l[0] * (c[0] + np.dot(c[-p:].T, zpn[p:0:-1]).T)

		# Compute the remaining causal polar coefficients
		for i in range(1, c.shape[0]):
			c[i, :] = 6. * c[i, :] + zp * c[i - 1, :]

		# Compute the initial anti-causal polar coefficient
		# c[-1] is never in the dot product since p < n
		c[-1] = l[1] * (c[-1] + np.dot(c[:p,].T, zpn[1:p+1]).T)

		# Compute the remaining anti-causal polar coefficients
		for i in reversed(range(c.shape[0] - 1)):
			c[i, :] = zp * (c[i + 1, :] - c[i,:])

		# Correct the length and coefficients for the azimuthal angle
		n, m, k = nphi, ntheta - 1, nphi / 2
		l = 6. / (1 - zpn[n]), zp / (zpn[n] - 1)

		# Limit the number of terms in the sum
		p = min(n - 1, self.precision)
		pk = min(k, self.precision)

		# The initial causal azimuthal coefficients from the second hemisphere
		c[1:m, 0] = l[0] * (c[1:m, 0] + np.dot(c[:-m:-1, -pk:], zpn[pk:0:-1]))
		# High precision may require terms from the first hemisphere
		if (p > k): c[1:m, 0] += l[0] * np.dot(c[1:m, k-p:], zpn[p:k:-1])

		# Compute the remaining coefficients of the first hemisphere
		for i in range(1, c.shape[1]):
			c[1:m, i] = 6. * c[1:m, i] + zp * c[1:m, i - 1]

		# Populate the initial coefficients of the second hemisphere
		c[:-m:-1, 0] = 6. * c[:-m:-1, 0] + zp * c[1:m, -1]

		# Compute the remaining coefficients of the second hemisphere
		for i in range(1, c.shape[1]):
			c[-m+1:, i] = 6. * c[-m+1:, i] + zp * c[-m+1:, i - 1]

		# The initial anti-causal azimuthal coefficients from the first hemisphere
		c[:-m:-1, -1] = l[1] * (c[:-m:-1, -1] + np.dot(c[1:m, :pk], zpn[1:pk+1]))
		# High precision may require terms from the second hemisphere
		if (p > k): c[:-m:-1, -1] += l[1] * np.dot(c[:-m:-1, :p-k], zpn[k+1:p+1])

		# Compute the remaining coefficients of the second hemisphere
		for i in reversed(range(c.shape[1] - 1)):
			c[-m+1:, i] = zp * (c[-m+1:, i + 1] - c[-m+1:, i])

		# Populate the initial coefficients of the first hemisphere
		c[1:m, -1] = zp * (c[:-m:-1, 0] - c[1:m, -1])

		# Compute the remaining coefficients of the first hemisphere
		for i in reversed(range(c.shape[1] - 1)):
			c[1:m, i] = zp * (c[1:m, i + 1] - c[1:m, i])

		# The polar azimuthal coefficients are special cases in which
		# the period degenerates to pi, rather than 2 pi.
		n = nphi / 2
		l = 6. / (1. - zpn[n]), zp / (zpn[n] - 1.)

		# Limit the number of terms in the sum
		p = min(n - 1, self.precision)

		# Compute the coefficients for each pole
		for i in [0, m]:
			# Compute the initial causal azimuthal coefficient
			c[i, 0] = l[0] * (c[i, 0] + np.dot(c[i, -p:], zpn[p:0:-1]))

			# Compute the remaining causal azimuthal coefficients
			for j in range(1, c.shape[1]):
				c[i, j] = 6. * c[i, j] + zp * c[i, j - 1]

			# Compute the initial anti-causal azimuthal coefficient
			c[i, -1] = l[1] * (c[i, -1] + np.dot(c[i, :p], zpn[1:p+1]))

			# Compute the remaining anti-causal azimuthal coefficients
			for j in reversed(range(c.shape[1] - 1)):
				c[i, j] = zp * (c[i, j + 1] - c[i, j])

		return c


	def weights(self, x):
		'''
		Evaluate the cubic b-spline interpolation weights for a
		fractional coordinate 0 <= x <= 1.
		'''

		tail = lambda y: (2. - abs(y))**3 / 6.
		hump = lambda y: (2. / 3.) - 0.5 * abs(y)**2 * (2 - abs(y))

		return [tail(1 + x), hump(x), hump(1 - x), tail(2 - x)]


	def interpolate(self, f):
		'''
		Given the angular function f, convert it to cubic b-spline
		coefficients and interpolate it on the previously defined grid.
		'''

		# Grab the cubic b-spline coefficients
		c = self.getcoeff(f)

		# Return the output
		return self.matrix * c.ravel('F')



class SphericalInterpolator(object):
	'''
	Build a sparse matrix that can be used to interpolate an angular
	function defined on the surface of a sphere.
	'''

	def __init__(self, thetas, order=4):
		'''
		Build the Lagrange interpolation matrix of a specified order
		for a regularly sampled angular function. Interpolation windows
		wrap around the pole.

		The 2-element list of lists thetas specifies the locations of
		polar samples for the coarse (thetas[0]) and fine (thetas[1])
		grids. The azimuthal samples at each of the levels have

			nphi[i] = 2 * (len(thetas[i]) - 2)

		regularly spaced values. The elements thetas[i][0] and
		thetas[i][-1] correspond to polar values that will only be
		sampled once.
		'''

		if order > len(thetas[0]):
			raise ValueError('Order should not exceed number of coarse samples.')

		# Grab the total number of polar samples
		ntheta = [len(t) for t in thetas]

		# Double the number of samples away form the poles
		nphi = [2 * (n - 2) for n in ntheta]
		# Don't duplicate azimuthal values at the poles
		nsamp = [2 + (nt - 2) * nph for nt, nph in zip(ntheta, nphi)]

		# Initialize the sparse matrix to copy the first polar value
		data = [1]
		ij = [[0, 0]]
		rval = 0

		# Grab the azimuthal step size
		dphi = [2 * math.pi / n for n in nphi]

		# Half the Lagrange interval width
		offset = (order - 1) / 2

		# Adjust wrapping shifts depending on direction of polar samples
		if thetas[0][0] < thetas[0][-1]: wraps = 0, 2 * math.pi
		else: wraps = 2 * math.pi, 0.

		# Adjust indices running off the right of the polar array
		tflip = lambda t: (ntheta[0] - t - 2) % ntheta[0]

		# Loop through all polar samples away from the poles
		for rtheta in thetas[1][1:-1]:
			# Find the starting interpolation interval
			tbase = cutil.rlocate(thetas[0], rtheta) - offset

			# Enumerate all polar indices involved in interpolation
			rows = [tbase + l for l in range(order)]

			# Build the corresponding angular positions
			tharr = []
			for i, ti in enumerate(rows):
				if ti < 0:
					tharr.append(wraps[0] - thetas[0][-ti])
				elif ti >= ntheta[0]:
					tharr.append(wraps[1] - thetas[0][tflip(ti)])
				else: tharr.append(thetas[0][ti])

			# Build the Lagrange interpolation coefficients
			twts = poly.lagrange(rtheta, tharr)

			# Loop over the higher azimuthal sampling rate
			for j in range(nphi[1]):
				# Increment the row pointer
				rval += 1
				# Take care of each theta sample
				for tw, rv in zip(twts, rows):
					# Pole samples are not azimuthally interpolated
					if rv == 0:
						data.append(tw)
						ij.append([rval, 0])
						continue
					elif rv == ntheta[0] - 1:
						data.append(tw)
						ij.append([rval, nsamp[0] - 1])
						continue

					# Compute the angular position
					rphi = j * dphi[1]
					# Find the starting interpolation interval
					k = (j * nphi[0]) / nphi[1] - offset

					if rv < 0 or rv >= ntheta[0]:
						rphi += math.pi
						k += nphi[0] / 2

					# Properly wrap the polar values
					if rv < 0: rv = -rv
					elif rv >= ntheta[0]: rv = tflip(rv)

					# Build the wrapped phi indices
					cols = [(k + m + nphi[0]) % nphi[0]
							for m in range(order)]
					# Build the unwrapped phi values
					pharr = [(k + m) * dphi[0] for m in range(order)]

					# Build the Lagrange interpolation coefficients
					pwts = poly.lagrange(rphi, pharr)

					# Populate the columns of the sparse array
					for pw, cv in zip(pwts, cols):
						vpos = 1 + (rv - 1) * nphi[0] + cv
						data.append(pw * tw)
						ij.append([rval, vpos])

		# Add the last pole value
		rval += 1
		data.append(1)
		ij.append([rval, nsamp[0] - 1])

		# Create a CSR matrix representation of the interpolator
		self.matrix = sparse.csr_matrix((data, zip(*ij)), shape=nsamp[::-1])


	def interpolate(self, f):
		'''
		Interpolate the coarsely sampled angular function f.
		'''
		# Compute the output
		return self.matrix * f


def polararray(ntheta, lobatto=True):
	'''
	Return a numpy array of polar angular samples.

	When lobatto is True, the samples correspond to Gauss-Lobatto quadrature
	points (including poles) in decreasing order.

	When lobatto is false, the samples are in increasing order at regular
	intervals that include the poles. Thus, the samples are

		theta = math.pi * np.arange(ntheta) / (ntheta - 1.).
	'''
	if not lobatto: return math.pi * np.arange(ntheta) / (ntheta - 1.)
	return quad.gausslob(ntheta)[0][::-1]

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
	ytlm = np.array([shxp(clm,poly.legassoc(deg-1,deg-1,tx)) for tx in t])

	# Return the product on the specified grid
	fld = np.tensordot(jlr, np.tensordot(ytlm, epm, axes=(1,0)), axes=(1,1))
	return fld.squeeze()


def translator (r, s, phi, theta, l):
	'''
	Compute the diagonal translator for a translation distance r, a
	translation direction s, azimuthal samples specified in the array phi,
	polar samples specified in the array theta, and a truncation point l.
	'''

	# The radial argument
	kr = 2. * math.pi * r

	# Compute the radial component
	hl = spec.sph_jn(l, kr)[0] + 1j * spec.sph_yn(l, kr)[0]
	# Multiply the radial component by scale factors in the translator
	m = numpy.arange(l + 1)
	hl *= (1j / 4. / math.pi) * (1j)**m * (2. * m + 1.)

	# Compute Legendre angle argument dot(s,sd) for sample directions sd
	stheta = numpy.sin(theta)[:,numpy.newaxis]
	sds = (s[0] * stheta * numpy.cos(phi)[numpy.newaxis,:]
			+ s[1] * stheta * numpy.sin(phi)[numpy.newaxis,:]
			+ s[2] * numpy.cos(theta)[:,numpy.newaxis])

	# Initialize the translator
	tr = 0

	# Sum the terms of the translator
	for hv, pv in izip(hl, poly.legpoly(sds, l)): tr += hv * pv
	return tr


def exband (a, tol = 1e-6):
	'''
	Compute the excess bandwidth estimation for an object with radius a
	to a tolerance tol.
	'''

	# Compute the number of desired accurate digits
	d0 = -math.log10(tol)

	# Return the estimated bandwidth
	return int(2. * math.pi * a + 1.8 * (d0**2 * 2. * math.pi * a)**(1./3.))


