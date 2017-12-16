'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''
# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np
import itertools
import functools

def sgimgcoeffs(img, *args, use_pyfftw=True, **kwargs):
	'''
	Given a 3-D image img with shape (nx, ny, nz), use Savitzky-Golay
	stencils from savgol(*args, **kwargs) to compute compute the filtered
	double-precision image coeffs with shape (nx, ny, nz, ns) such that
	coeffs[:,:,:,i] holds the convolution of img with the i-th stencil.

	If the image is of single precision, the filter correlation will be done
	in single-precision; otherwise, double precision will be used.

	If the keyword-only argument use_pyfftw is True, the pyfftw module will
	be used (if available) to accelerate FFT correlations. Otherwise, the
	stock Numpy FFT will be used. The argument is not passed to savgol.
	'''
	# Create the stencils first
	stencils = savgol(*args, **kwargs)
	if not stencils: raise ValueError('Savitzky-Golay stencil list is empty')

	# Make sure the array is in double precision
	img = np.asarray(img)
	if img.ndim != 3: raise ValueError('Image img must be three-dimensional')

	# If possible, find the next-larger efficient size
	try: from scipy.fftpack.helper import next_fast_len
	except ImportError: next_fast_len = lambda x: x

	# Half-sizes of kernels along each axis
	hsizes = tuple(bsz // 2 for bsz in stencils[0].shape)

	# Padded shape for FFT convolution and the R2C FFT output
	pshape = tuple(next_fast_len(isz + 2 * bsz)
			for isz, bsz in zip(img.shape, hsizes))

	if img.dtype == np.dtype('float32'):
		ftype, ctype = np.dtype('float32'), np.dtype('complex64')
	else:
		ftype, ctype = np.dtype('float64'), np.dtype('complex128')

	try:
		if not use_pyfftw: raise ImportError
		import pyfftw
	except ImportError:
		from numpy.fft import rfftn, irfftn
		empty = np.empty
		use_fftw = False
	else:
		# Cache PyFFTW planning for 5 seconds
		empty = pyfftw.empty_aligned
		use_fftw = True

	# Build working and output arrays
	kernel = empty(pshape, dtype=ftype)
	output = empty(img.shape + (len(stencils),), dtype=ftype)

	if use_fftw:
		# Need to create output arrays and plan both FFTs
		krfft = empty(pshape[:-1] + (pshape[-1] // 2 + 1,), dtype=ctype)
		rfftn = pyfftw.FFTW(kernel, krfft, axes=(0, 1, 2))
		irfftn = pyfftw.FFTW(krfft, kernel,
				axes=(0, 1, 2), direction='FFTW_BACKWARD')

	m,n,p = img.shape

	# Copy the image, leaving space for boundaries
	kernel[:,:,:] = 0.
	kernel[:m,:n,:p] = img

	# For right boundaries, watch for running off left end with small arrays
	for ax, (ld, hl)  in enumerate(zip(img.shape, hsizes)):
		# Build the slice for boundary values
		lslices = [slice(None)]*3
		rslices = [slice(None)]*3

		# Left boundaries are straightforward
		lslices[ax] = slice(hl, 0, -1)
		rslices[ax] = slice(-hl, None)
		kernel[rslices] = kernel[lslices]

		# Don't walk off left edge when mirroring right boundary
		hi = ld - 1
		lo = max(hi - hl, 0)
		lslices[ax] = slice(lo, hi)
		rslices[ax] = slice(2 * hi - lo, hi, -1)
		kernel[rslices] = kernel[lslices]

	# Compute the image FFT
	if use_fftw:
		rfftn.execute()
		imfft = krfft.copy()
	else: imfft = rfftn(kernel)

	i,j,k = hsizes
	t,u,v = stencils[0].shape

	for l, stencil in enumerate(stencils):
		# Clear the kernel storage and copy the stencil
		kernel[:,:,:] = 0.
		kernel[:t,:u,:v] = stencil[::-1,::-1,::-1]
		if use_fftw:
			rfftn.execute()
			krfft[:,:,:] *= imfft
			irfftn(normalise_idft=True)
		else: kernel = irfftn(rfftn(kernel) * imfft)
		output[:,:,:,l] = kernel[i:i+m,j:j+n,k:k+p]

	return output


@functools.lru_cache(maxsize=32)
def savgol(size, order=2):
	'''
	Compute a Savitzky-Golay filter for a cubic tile of width size with
	the given interpolating order. The tile size must be odd and must
	satisfy size > order.

	The return value is a four-tuple of filter stencils (b, bx, by, bz),
	where convolution of an image with b yields the filtered image and
	convolution of the image with bx, by or bz yields the x, y or z
	derivatives of the filtered image, respectively.

	This function is memoized with functools.lru_cache; it will save the
	results of the most recent 32 calls for efficiency.
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
