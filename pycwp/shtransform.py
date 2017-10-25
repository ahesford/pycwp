# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy, math, numpy.fft as fft

from scipy import special as spec

from . import cutil, quad, poly

class SHTransform(object):
	'''
	Encapsulates a spherical harmonic transform to convert between
	spherical harmonic coefficients and plane-wave coefficients.
	'''
	def __init__ (self, deg, ntheta = 0, nphi = 0):
		'''
		Establishes a harmonic transform between coefficients of
		maximum degree deg with ntheta polar and nphi azimuthal
		angular samples.
		'''
		# Copy over the relevant parameters
		self.deg, self.ntheta, self.nphi = deg, ntheta, nphi

		# Set some minimum values if necessary
		if self.ntheta < deg:
			self.ntheta = deg
		if self.nphi < 2 * deg - 1: 
			self.nphi = 2 * deg - 1

		# Allocate a list of the polar samples and integration weights
		self.thetas, self.weights = quad.gaussleg (self.ntheta)

	def scale (self, shdata, forward=True):
		'''
		Scale the spherical harmonic coefficients to relate the
		far-field signature of the feld. The default forward
		direction properly scales coefficients AFTER a forward
		transform (angular to spherical). If forward is false,
		scale the coefficients before an INVERSE transform.
		'''
		# Set up a generator for the scale factor
		if forward:
			cscale = (1j**(i % 4) for i in range(1,self.deg + 1))
		else:
			cscale = ((-1j)**(i % 4) for i in range(1,self.deg + 1))

		# Scale each column of the coefficient matrix
		for column in shdata[:,:self.deg].transpose():
			column *= next(cscale)

	def spectogrid (self, samples):
		'''Execute an inverse harmonic transform.'''

		# Build the output array
		output = numpy.zeros (samples.shape, dtype=complex)
		# Get transposed views for iterating over columns
		outtr = output.transpose()
		samptr = samples.transpose()

		for theta, out in zip (self.thetas, outtr):
			# Compute the normalized associated Legendre polynomials
			legpol = poly.legassoc(self.deg-1, self.deg-1, theta)
			# Transpose the matrix for iterations
			legpol = legpol.transpose()

			for deg, (pol, smp) in enumerate(zip(legpol, samptr)):
				# Handle the positive orders
				out[:deg+1] += pol[:deg+1] * smp[:deg+1]
				# Handle the negative orders
				out[self.nphi-deg:] += pol[deg:0:-1] * smp[self.nphi-deg:]

		return self.nphi * fft.ifft (output, axis=0)

	def gridtospec (self, samples):
		'''Execute a forward harmonic transform.'''

		# Build the output array
		output = numpy.zeros (samples.shape, dtype=complex)
		# Get transposed views for iterating over columns
		outtr = output.transpose()
		# Take the Fourier transform of the input and transpose
		samptr = fft.fft (samples, axis=0).transpose()

		for theta, weight, smp in zip(self.thetas, self.weights, samptr):
			scale = 2. * math.pi * weight / self.nphi

			# Compute the associated Legendre polynomials
			legpol = poly.legassoc(self.deg-1, self.deg-1, theta)
			# Transpose the matrix for iterations
			legpol = legpol.transpose()

			# Update the output array
			for deg, (out, pol) in enumerate(zip(outtr, legpol)):
				# Handle the positive orders
				out[:deg+1] += pol[:deg+1] * smp[:deg+1] * scale
				# Handle the negative orders
				out[self.nphi-deg:] += scale * pol[deg:0:-1] * smp[self.nphi-deg:]

		return output

def filter (infield, maxord):
	'''
	Set the harmonic order and sample size of the input field to maxord.
	Output array is in FORTRAN order.
	'''

	# Make sure the input matrix is the proper shape and size
	if infield.shape[0] < 2 * infield.shape[1] - 1:
		raise IndexError ('input matirx not the right shape')
	
	# Preallocate the array
	outfield = numpy.zeros ((2 * maxord - 1, maxord), dtype=complex)

	# Find the number of terms to copy
	nterm = min (infield.shape[1], maxord)

	# Copy the positive-order terms
	outfield[0:nterm,0:nterm] = infield[0:nterm,0:nterm]
	# Copy the negative-order terms
	outfield[-nterm+1:,0:nterm] = infield[-nterm+1:,0:nterm]

	return outfield
