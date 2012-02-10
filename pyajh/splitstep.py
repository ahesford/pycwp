'''
Routines to compute split-step propagation through slices of media.
'''

import math, numpy as np, numpy.fft as fft

from . import mio

class SplitStepEngine(object):
	'''
	A class to advance a wavefront through an arbitrary medium according to
	the split-step method.
	'''

	def __init__(self, k0, nx, ny, h):
		'''
		Initialize a split-step engine over an nx-by-ny grid with
		isotropic step size h. The unitless wave number is k0.
		'''

		# Copy the parameters
		self.grid = nx, ny
		self.h, self.k0 = h, k0

		# The default propagator and attenuation screen are None
		self._propagator = None
		self._attenuator = None


	def kxysq(self):
		'''
		Return the square of the transverse wave number, caching the
		result if it has not already been computed.
		'''

		try: return self._kxysq
		except AttributeError:
			# Get the coordinate indices for each slab
			nx, ny = self.grid
			sl = np.mgrid[-nx/2:nx/2, -ny/2:ny/2].astype(float)

			# Compute the transverse wave numbers
			kx = 2. * math.pi * sl[0] / (nx * self.h)
			ky = 2. * math.pi * sl[1] / (ny * self.h)

			self._kxysq = fft.fftshift(kx**2 + ky**2)
			return self._kxysq 


	def slicecoords(self):
		'''
		Return the meshgrid coordinate arrays for an x-y slab in the
		current engine. The origin is in the center of the slab.
		'''
		g = self.grid
		cg = np.mgrid[:g[0], :g[1]].astype(float)
		return [(c - 0.5 * (n - 1.)) * self.h for c, n in zip(cg, g)]


	@property
	def propagator(self):
		'''
		Return a propagator to advance wabes to the next slab.
		'''
		return self._propagator


	@propagator.setter
	def propagator(self, dz):
		'''
		Generate and cache the slab propagator for a step size dz. If
		dz is None, use a step that matches the transverse step.
		'''
		# Make the grid isotropic if slab thickness isn't specified
		if dz is None: dz = self.h

		# Get the longitudinal wave number
		kz = np.sqrt(complex(self.k0**2) - self.kxysq())

		# Compute the propagator
		self._propagator = np.exp(1j * kz * dz)


	@propagator.deleter
	def propagator(self): del self._propagator


	def propfield(self, fld):
		'''
		Apply the spectral propagator to the provided field.
		'''
		return fft.ifftn(self.propagator * fft.fftn(fld))


	@property
	def attenuator(self):
		'''
		Return a screen to attenuate the field at each slab edge. If
		one was not previously set, a screen that does not attenuate is
		created.
		'''
		return self._attenuator


	@attenuator.setter
	def attenuator(self, aparm):
		'''
		Generate a screen to supresses the field in each slab. The
		tuple aparm takes the form (a, t), where a is the maximum
		attenuation exp(-a), and t is the thickness of the border over
		which the attenuation increases from zero to the maximum.
		'''
		# Unfold the parameter tuple
		atten, nt = aparm
		# Build an index grid centered on the origin
		nx, ny = self.grid
		sl = np.mgrid[:nx,:ny].astype(float)
		sl[0] -= (nx - 1.) / 2.
		sl[1] -= (ny - 1.) / 2.
		# Positive points lie within the absorbing boundaries
		x = (np.abs(sl[0]) - (0.5 * nx - nt)) / float(nt)
		y = (np.abs(sl[1]) - (0.5 * ny - nt)) / float(nt)

		# Compute the attenuation profile
		wx = 1 - np.sin(0.5 * math.pi * (x > 0).choose(0, x))**2
		wy = 1 - np.sin(0.5 * math.pi * (y > 0).choose(0, y))**2
		w = wx * wy

		# This is the imaginary part of the wave number in the boundary
		self._attenuator = 1j * atten * (1. - w) / self.k0


	@attenuator.deleter
	def attenuator(self): del self._attenuator


	def phasescreen(self, eta, dz = None):
		'''
		Generate the phase screen for a slab characterized by scaled
		wave number eta. If the step size dz is not specified, the
		isotropic in-plane step is used. Results are not cached.
		'''

		# Use the default step size if necessary
		if dz is None: dz = self.h

		return np.exp(1j * dz * self.k0 * (eta - 1.))


	def wideangle(self, fld, eta, dz = None):
		'''
		Apply wide-angle corrections to the propagated field fld
		corresponding to a medium with scaled wave number eta.
		'''
		if dz is None: dz = self.h

		# Compute the Laplacian term and the spatial correction
		kbar = self.kxysq() / self.k0**2
		cor = 1j * self.k0 * dz * (eta - 1.) / (2. * eta)
		return fld + cor * fft.ifftn(kbar * fft.fftn(fld))


	def advance(self, fld, obj, dz = None):
		'''
		Use the spectral propagator, the phase screen for the current
		slab, and the attenuation screen to advance the field through
		one layer of a specified medium with contrast obj. The
		propagator and attenuation screen must have previously been
		established.
		'''

		# Convert the contrast to a scaled wave number
		eta = np.sqrt(obj + 1.)

		# Incorporate an absorbing boundary if one was desired
		if self.attenuator is not None: eta += self.attenuator

		# Build the phase screen for this slab
		screen = self.phasescreen(eta, dz)

		# Propagate the field with a wide-angle term and phase screen
		fld = screen * self.wideangle(self.propfield(fld), eta, dz)

		return fld
