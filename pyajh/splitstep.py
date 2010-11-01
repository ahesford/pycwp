'''
Routines to compute split-step propagation through slices of media.
'''

import numpy as np
import numpy.fft as fft
import math

def slicecoords(nx, ny, dx):
	'''
	Return the meshgrid arrays x and y given by an (nx,ny) grid with
	uniform spacing dx.
	'''

	slice = np.mgrid[-nx/2:nx/2,-ny/2:ny/2] * float(dx)
	return slice[0], slice[1]

def k(f, c):
	'''
	Compute the wave number for a frequency f and a sound speed c.
	'''
	return 2.0 * math.pi * f / c

def incslab(k0, x, y, zh, src):
	'''
	Compute an incident field with wave number k0 due to a source at
	location src in the slab at height zh with transverse coordinates
	specified in the meshgrid arrays x and y.
	'''

	rhosq = (x - src[0])**2 + (y - src[1])**2
	rsq = rhosq + (zh - src[2])**2
	r = np.sqrt(rsq)

	return np.exp(1j * k0 * r - 2 * rhosq / rsq) / r

def genprop(k0, nx, ny, dx, dz = None):
	'''
	Generate the propagator used to advance the field through a homogeneous
	slab of dimensions (nx, ny) and with height dx and wave number k0.
	'''

	# Make the grid isotropic if slab thickness isn't specified
	if dz is None: dz = dx

	# Get the coordinate indices for the slab
	slice = np.mgrid[-nx/2:nx/2,-ny/2:ny/2]

	# Compute the transverse wave numbers
	kx = 2. * math.pi * slice[0] / (nx * dx)
	ky = 2. * math.pi * slice[1] / (ny * dx)
	ksq = kx**2 + ky**2

	# Return the propagator, with frequencies shifted for proper alignment
	return fft.fftshift(np.exp(1j * np.sqrt(complex(k0)**2 - ksq) * float(dz)))

def genatten(atten, nt, nx, ny):
	'''
	Generate an attenuation screen that supresses the field in a slab of
	size (nx, ny) for a thickness of nt points around each edge.
	'''

	slice = np.mgrid[-nx/2:nx/2,-ny/2:ny/2]

	# Figure out how much attenuation should be applied in the slab
	x = (np.abs(slice[0]) - (nx / 2. - nt)) / float(nt)
	y = (np.abs(slice[1]) - (ny / 2. - nt)) / float(nt)

	# Compute the attenuation profile
	wx = 1 - np.sin(0.5 * math.pi * (x > 0).choose(0, x))**2
	wy = 1 - np.sin(0.5 * math.pi * (y > 0).choose(0, y))**2
	w = wx * wy;

	return np.exp(-atten * (1. - w))

def genscreen(obj, k0, dx):
	'''
	Generate the phase screen used to correct for differences from the
	average wave number. The background wave number k0 should be unitless,
	and the object contrast obj is the square of the ratio of the true wave
	number to the background wave number, minus 1. The slab thickness dx
	should be in wavelengths.
	'''
	return np.exp(0.5j * float(dx) * k0 * obj)

def propfield(fld, prop):
	'''
	Apply the propagator in the spatial-frequency domain.
	'''

	return fft.ifftn(prop * fft.fftn(fld))
