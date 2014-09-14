'''
These routines convert a segmented tissue model based on MRI data into acoustic
parameters of sound speed, attenuation and density for simulation of ultrasound
wave propagation.
'''

import numpy as np, math
from scipy import ndimage
from . import mio, cutil


def randparam(scat, mean, stdfrac):
	'''
	Given the smoothed scatter matrix scat returned by scatsmear, return a
	parameter map with the specified mean and standard deviation (specified
	as a fraction of the mean value).

	If stdfrac is None, no random variations are assumed and the mean value
	is returned as a scalar.
	'''
	if stdfrac is None: return mean
	else: return (stdfrac * mean) * scat + mean


def bumpmap(ptden, sigma, shape):
	'''
	Return a matrix with the specified shape that corresponds to randomly
	placed Gaussian scatterers with a fractional density ptden and a
	standard deviation sigma in pixels.

	The resulting distribution has zero mean and unit standard deviation.
	'''
	ntot = cutil.prod(shape)
	# Compute the number of points that will be nonzero
	npts = ptden * ntot
	# Compute the locations of the nonzero points
	idx = (np.random.rand(npts) * ntot).astype(int)
	# Set the nonzero locations
	scat = np.zeros(ntot)
	scat[idx] = 1.
	scat = scat.reshape(shape, order='F')

	# Filter the center locations
	scat = ndimage.gaussian_filter(scat, sigma, mode='constant')
	# Normalize the distribution
	scat -= np.mean(scat)
	scat /= np.std(scat)
	return scat


def maptissueblk(seg, params, n, scatden=0.2, scatsd=0.6, chunk=8, smoothp=[5,3]):
	'''
	Given a segmentation description in the Slicer seg, compute the sound
	speed, attenuation and density (with random variations) of the block
	with chunk slices starting at slice n.

	The material parameters are specified in

		params = [ soundparams, atnparams, denparams ],

	where each individual item is a list of the form

		soundparams[type] = [ mean, stdfrac ],

	where mean is the mean parameter value and stdfrac is the standard
	deviation specified as a fraction of the mean. The index type is the
	value of the tissue type in the segmentation file.

	If the stdpct value is None for a particular tissue type, then no
	random variations are added to the mean tissue parameter.

	Centers of random scatters are positioned with fractional density
	scatden and are characterized by Gaussian variations with a standard
	deviation of scatsd voxels.

	The list smoothp specifies the width and standard deviation in pixels,
	respectively, of the Gaussian pulse used to smooth tissue boundaries.
	'''
	# Build the Gaussian kernel for smoothing the tissue map
	kern = cutil.smoothkern(*smoothp, n = 1)
	# This convenience function smooths the tissue masks
	def smooth3(a):
		b = ndimage.correlate1d(a, kern, mode='nearest', axis=0)
		c = ndimage.correlate1d(b, kern, mode='nearest', axis=1)
		# This avoids the need to allocate another temporary buffer
		ndimage.correlate1d(c, kern, output=b, mode='nearest', axis=2)
		return b
	# Pad the blocks with the kernel width for valid convolutions
	pad = (smoothp[0] - 1) / 2

	# Try to start two slices early for correct convolution
	start = max(0, n - pad)
	# Try to read two extra slices for correct convolution
	finish = min(n + chunk + pad, seg.shape[-1])
	# Read the correctly sized data block
	block = seg[start:finish].astype(int)
	# Create the sound speed, attenuation and density blocks
	snd = np.zeros(block.shape, dtype=np.float32, order='F')
	atn = np.zeros(block.shape, dtype=np.float32, order='F')
	den = np.zeros(block.shape, dtype=np.float32, order='F')

	# Create the random bump map
	bumps = bumpmap(scatden, scatsd, block.shape)

	# Loop through each of the tissue types
	for k, (sp, ap, dp) in enumerate(zip(*params)):
		# Smooth the mask
		mask = smooth3((block == k).astype(float))
		# Add the parameters for this tissue
		snd += mask * randparam(bumps, *sp)
		atn += mask * randparam(bumps, *ap)
		den += mask * randparam(bumps, *dp)

	# Compute the start and end slices of the parameters
	istart = n - start
	iend = min(chunk + istart, snd.shape[-1])

	# Return the appropriately sized chunk
	return snd[:,:,istart:iend], atn[:,:,istart:iend], den[:,:,istart:iend]
