'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''
# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np

import itertools

def sgimgcoeffs(img, *args, **kwargs):
	'''
	Given a 3-D image img with shape (nx, ny, nz), use Savitzky-Golay
	stencils from savgol(*args, **kwargs) to compute compute the filtered
	double-precision image coeffs with shape (nx, ny, nz, ns) such that
	coeffs[:,:,:,i] holds the convolution of img with the i-th stencil.
	'''
	# Create the stencils first
	stencils = savgol(*args, **kwargs)
	if not stencils: raise ValueError('Savitzky-Golay stencil list is empty')

	# Make sure the array is in double precision
	img = np.asarray(img)
	if img.ndim != 3: raise ValueError('Image img must be three-dimensional')

	try:
		import pyfftw
		from pyfftw.interfaces.numpy_fft import rfftn, irfftn
	except ImportError:
		from numpy.fft import rfftn, irfftn
		empty = np.empty
	else:
		# Cache PyFFTW planning for 5 seconds
		pyfftw.interfaces.cache.enable()
		pyfftw.interfaces.cache.set_keepalive_time(5.0)
		empty = pyfftw.empty_aligned

	# If possible, find the next-larger efficient size
	try: from scipy.fftpack.helper import next_fast_len
	except ImportError: next_fast_len = lambda x: x

	# Half-sizes of kernels along each axis
	hsizes = tuple(bsz // 2 for bsz in stencils[0].shape)

	# Padded shape for FFT convolution and the R2C FFT output
	pshape = tuple(next_fast_len(isz + 2 * bsz)
			for isz, bsz in zip(img.shape, hsizes))
	rshape = pshape[:-1] + (pshape[-1] // 2 + 1,)

	# Build working and output arrays
	kernel = empty(pshape, dtype=np.float64)
	output = empty(img.shape + (len(stencils),), dtype=np.float64)

	i,j,k = hsizes
	m,n,p = img.shape
	t,u,v = stencils[0].shape

	# Copy the image, leaving space for boundaries
	kernel[:,:,:] = 0.
	# Ignore extra padding that may have been added for efficiency
	kernel[:m,:n,:p] = img
	# Mirror left boundaries
	kernel[-i:,:,:] = kernel[i:0:-1,:,:]
	kernel[:,-j:,:] = kernel[:,j:0:-1,:]
	kernel[:,:,-k:] = kernel[:,:,k:0:-1]
	# Right boundaries (double indexing avoids negative index problems)
	if m > i: kernel[m:m+i,:,:] = kernel[m-i-1:m-1,:,:][::-1,:,:]
	if n > j: kernel[:,n:n+j,:] = kernel[:,n-j-1:n-1,:][:,::-1,:]
	if p > k: kernel[:,:,p:p+k] = kernel[:,:,p-k-1:p-1][:,:,::-1]

	# Compute the image FFT
	imfft = rfftn(kernel)

	for l, stencil in enumerate(stencils):
		# Clear the kernel storage and copy the stencil
		kernel[:,:,:] = 0.
		kernel[:t,:u,:v] = stencil[::-1,::-1,::-1]
		output[:,:,:,l] = irfftn(rfftn(kernel) * imfft)[i:i+m,j:j+n,k:k+p]

	return output


def savgol(size, order=2):
	'''
	Compute a Savitzky-Golay filter for a cubic tile of width size with
	the given interpolating order. The tile size must be odd and must
	satisfy size > order.

	The return value is a four-tuple of filter stencils (b, bx, by, bz),
	where convolution of an image with b yields the filtered image and
	convolution of the image with bx, by or bz yields the x, y or z
	derivatives of the filtered image, respectively.
	'''
	size = int(size)
	if size <= order or not size % 2:
		raise ValueError('Tile size size must be odd and exceed order')

	# Convert width to half-width
	n = size // 2

	a = [ ]

	for x, y, z in itertools.product(range(-n, n+1), repeat=3):
		ar = [ ]
		for i, j, k in itertools.product(range(order+1), repeat=3):
			ar.append(x**i * y**j * z**k)
		a.append(ar)

	b = np.linalg.pinv(a)

	# Row indices of the function and first-derivative filters
	dfidx = [ 0, (order + 1)**2, (order + 1), 1 ]
	return tuple(b[r].reshape((2 * n + 1,)*3, order='C') for r in dfidx)
