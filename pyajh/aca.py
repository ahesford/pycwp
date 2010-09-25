import numpy as np
import math

from itertools import izip

def matrix(eltfunc, nr, nc, tol = 1e-6):
	'''
	Compute the ACA approximation, with tolerance tol, to a matrix with nr
	rows and nc columns. The function eltfunc(i,j) returns the matrix
	element for the i-th row and j-th column, starting from 0.
	'''

	# Work with the square of the tolerance
	tol = tol**2

	# Find the maximum rank of the matrix
	maxrank = min(nr, nc)

	# Initialize the list of used rows and columns
	irow, icol = [0], [] 

	u = []
	v = []
	zabs = 0.
	zhist = []

	# Loop through the algorithm
	for k in xrange(maxrank):
		# Build the desired row
		r = np.array([eltfunc(irow[-1], j) for j in xrange(nc)])

		# Subtract off existing contributions
		for ue, ve in izip(u, v):
			r -= ue[irow[-1]] * ve

		# Find the column index of the largest element
		icol.append(max([((idx in icol and [0.0] or [abs(l)])[0], idx)
			for idx, l in enumerate(r)])[-1])

		# Add in the scaled new row
		v.append(r / r[icol[-1]])

		# Build the desired column
		c = np.array([eltfunc(i, icol[-1]) for i in xrange(nr)])

		# Subract off existing contributions
		for ue, ve in izip(u, v):
			c -= ve[icol[-1]] * ue

		# Add in the new column
		u.append(c)

		# Find the row index of the largest element
		irow.append(max([((idx in irow and [0.0] or [abs(l)])[0], idx)
			for idx, l in enumerate(c)])[-1])

		# Update the new matrix norm estimate
		nrmsq = np.dot(np.conj(u[-1]), u[-1]) * np.dot(np.conj(v[-1]), v[-1])
		zabs += nrmsq

		# Add in the norm contributions from other row/column products
		for ue, ve in izip(u[:-1], v[:-1]):
			zabs += 2. * (np.abs(np.dot(ue, u[-1])) * 
					np.abs(np.dot(ve, v[-1])))

		if nrmsq <= tol * zabs: break

		# Store the convergence history of the ACA
		zhist.append(math.sqrt(nrmsq / zabs))

	return np.array(u), np.array(v), zhist
