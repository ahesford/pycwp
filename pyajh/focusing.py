'''
Routines for dealing with focused beams in three dimensions.
'''

import numpy
import math
import numpy.fft as fft

import scipy.interpolate as intp

from pyajh import cutil

def focusedbeam (f, c0, w, x_f, shft = 0.0, z_off = 0.0):
	'''
	Compute the coefficients for an elevation-focused beam with a the
	following paramters:

		f:     Frequency (Hz)
		c0:    Background sound speed (m/s)
		w:     Aperture width (m)
		x_f:   Focus length (m)
		shft:  A transverse focus shift (m) [Default: 0]
		z_off: Height offset (m) [Default: 0]
	'''
	# Wave number and wavelength
	k0 = 2.0 * math.pi * f / c0
	wl = c0 / f
	
	# Elevation spatial sampling frequency
	nusz = 2.0 * f / c0
	
	# Sample spacing in z
	delz = 1 / nusz
	
	# The number of samples to use in z
	nz = math.trunc (4.0 * (w + abs(z_off)) * nusz)

	# SD of Gaussian amplitude
	sigma = w / (2.0 * math.sqrt (math.pi));
	
	# Find the largest power of 2 greater than nz to optimize the FFT
	nz = 2**math.ceil(math.log(nz,2))
	
	# The spatial frequency vector
	vz = numpy.arange (-nz/2, nz/2) * nusz / nz
	# The spatial sampling vector
	za = numpy.arange (-nz/2, nz/2) * delz

	# Build the spatial samples: phase first
	t = numpy.exp(-1j * k0 * numpy.sqrt(x_f**2 + (za - z_off)**2))
	# Build the spatial samples: now the envelope
	t *= delz * math.sqrt(2.0) * numpy.exp(-(za - z_off)**2 / (2 * sigma**2)) 

	# Find the Fourier transform
	T = fft.fftshift (fft.fft (fft.fftshift (t)))

	# Use a propagation phase factor
	prop = numpy.exp (1j * 2.0 * math.pi *
			(x_f + shft) * numpy.sqrt((f/c0)**2 - vz**2))
	# Zero-out the evanescent parts
	prop = (abs(vz) >= abs(f/c0)).choose(prop,0)

	# Apply the masked propagation factor
	T *= prop

	# Record the angular samples
	theta = [math.acos (wl * vze) for vze in vz]
	
	return (theta, T)

def recvfocus (mat, thr, coeffs, theta):
	'''
	Apply receive focusing to a 3-D samples, with polar angles given 
	by thr, to generate a 2-D pattern. The focusing coefficients are
	defined over the polar angles theta.
	'''

	# Reverse the coefficients if theta is non-increasing.
	if theta[1] < theta[0]:
		theta.reverse()
		coeffs.reverse()

	# Build the interpolation function.
	cfunc = intp.interp1d (theta, coeffs, kind='quadratic',
			bounds_error=False, fill_value=0.+0j)

	# Compute the new coefficients.
	nc = cfunc (thr)

	# Return the focused array.
	return numpy.dot(mat, nc)

def focusfield (k, x, z, coeffs, theta, normalize=False):
	'''
	Compute the field for plane waves with amplitudes given by coeffs,
	elevation angles given by theta, wave number k, transverse position
	x and elevation position z. Optionally normalize the field.
	'''

	# Set up the blank field storage
	fld = numpy.zeros(x.shape, dtype=complex)

	# Add each plane wave to the overall field
	for (c, t) in zip(coeffs, theta):
		fld += c * numpy.exp(1j * k * (x * math.sin(t) + z * math.cos(t)))

	# Normalize the field if desired
	if normalize:
		fld /= cutil.complexmax(fld)

	return fld
