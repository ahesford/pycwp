'''
These routines convert a segmented tissue model based on MRI data into acoustic
parameters of sound speed, attenuation and density for simulation of ultrasound
wave propagation.
'''

import numpy as np, math
from scipy import ndimage
from . import mio


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
	ntot = np.prod(shape)
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


def smooth3(a, w, s):
	'''
	Mimic the behavior of the MATLAB function smooth3, which convolves the
	matrix a with a Gaussian kernel of width w and standard deviation s (in
	voxels). This is useful if the width is less than four standard
	deviations, in each direction, which is the enforced width with
	ndimage.gaussian_filter.
	'''
	if w % 2 != 1: raise ValueError('Kernel width must be odd.')
	lw = (w - 1) / 2
	# Compute the restricted Gaussian kernel
	k = np.zeros([w]*3)
	k[lw,lw,lw] = 1.
	k = ndimage.gaussian_filter(k, s, mode='constant')
	k /= np.sum(k)
	# Now perform the convolution
	# Note that correlation is the same because the kernel is symmetric
	return ndimage.correlate(a, k, mode='nearest')


def maptissue(segfile, outfiles, params, scatden=0.2, scatsd=0.6, chunk=8, slices=None):
	'''
	Given a segmentation description in segfile, compute the sound speed,
	attenuation and density (with random variations), writing the results
	to the sound speed, attenuation and density files

		outfiles = [ soundfile, attenfile, denfile ].

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

	If slices is not None, it takes the form [start, end] and indicates that
	the Python slice range start:end of the segmentation file and outputs
	should be processed. Otherwise, the entire file is processed.

	The output is processed in chunk slices at a time.
	'''
	# Open the segmentation file
	seg = mio.Slicer(segfile)
	# Open the output files, which should already have been created
	snd, atn, den = [mio.Slicer(f) for f in outfiles]
	# Check that the sizes all agree
	if list(snd.shape) != list(seg.shape):
		raise ValueError('The dimensions of the sound file '
				'do not match those of the segmentation file')
	if list(atn.shape) != list(seg.shape):
		raise ValueError('The dimensions of the attenuation file '
				'do not match those of the segmentation file')
	if list(den.shape) != list(seg.shape):
		raise ValueError('The dimensions of the density file '
				'do not match those of the segmentation file')
	# Process the whole file by default
	if slices is None: slices = [0, seg.shape[-1]]
	# Parameters for Gaussian smoothing of the tissue map
	smoothp = [5, 3]
	# The blocks have to be padded for valid convolutional smoothing
	pad = (smoothp[0] - 1) / 2

	# Process each chunk
	for n in range(slices[0], slices[1], chunk):
		print 'Processing chunk', n
		# Try to start two slices early for correct convolution
		start = max(0, n - pad)
		# Try to read two extra slices for correct convolution
		finish = min(n + chunk + pad, seg.shape[-1])
		# Read the correctly sized data block
		block = seg[start:finish].astype(int)
		# Create the sound speed, attenuation and density blocks
		SoundSpeed = np.zeros(block.shape, dtype=snd.dtype, order='F')
		Attenuation = np.zeros(block.shape, dtype=atn.dtype, order='F')
		Density = np.zeros(block.shape, dtype=den.dtype, order='F')

		# Create the random bump map
		bumps = bumpmap(scatden, scatsd, block.shape)

		# Loop through each of the tissue types
		for k, (sp, ap, dp) in enumerate(zip(*params)):
			# Smooth the mask
			mask = smooth3((block == k).astype(float), *smoothp)
			# Add the parameters for this tissue
			SoundSpeed += mask * randparam(bumps, *sp)
			Attenuation += mask * randparam(bumps, *ap)
			Density += mask * randparam(bumps, *dp)

		# Compute the start and end slices of the parameters
		istart = pad if n != 0 else 0
		iend = min(chunk + istart, SoundSpeed.shape[-1])
		# Compute the start and end slices of the files
		oend = min(seg.shape[-1], n + iend - istart)

		# Write the blocks to the outputs
		snd[n:oend] = SoundSpeed[:,:,istart:iend]
		atn[n:oend] = Attenuation[:,:,istart:iend]
		den[n:oend] = Density[:,:,istart:iend]
