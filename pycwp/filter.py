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
	img = np.asarray(img, dtype=np.float64)
	if img.ndim != 3:
		raise ValueError('Image img must be three-dimensional')

	try:
		import pyfftw
	except ImportError:
		from scipy.fftpack import fftn, ifftn
	else:
		fftn = pyfftw.interfaces.scipy_fftpack.fftn
		ifftn = pyfftw.interfaces.scipy_fftpack.ifftn
		
		pyfftw.interfaces.cache.enable()
		pyfftw.interfaces.cache.set_keepalive_time(5.)

	from scipy.signal import correlate

	# Use unfiltered image near boundary, but let derivatives roll off smoothly
	coeffs = np.zeros(img.shape + (len(stencils),), dtype=np.float64)
	coeffs[:,:,:,0] = img

	# If image is smaller than stencil, it is all boundary
	if any(bsz > isz for isz, bsz in zip(img.shape, stencils[0].shape)):
		return coeffs
	
	for l, ba in enumerate(stencils):
		cimg = correlate(img, ba, mode='valid')
		mx, my, mz = cimg.shape
		lx, ly, lz = ((nv - mv) // 2 for nv, mv in zip(img.shape, cimg.shape))
		coeffs[lx:lx+mx,ly:ly+my,lz:lz+mz,l] = cimg

	return coeffs


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
