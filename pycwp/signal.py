'''
Routines used for manipulation of sequences representing signals.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy, math
from numpy import fft
from itertools import izip

def findpeaks(vec, minwidth=None, minprom=None, prommode='absolute'):
	'''
	Find all peaks in the 1-D sequence vec. Return a list with elements
	(index, value, width, prominence) for a peak at the indicated index
	such that vec[index] == value, width is the distance from the peak to
	its key col, and prominence is the peak's height above its key col.

	Only peaks with widths greater than a specified minwidth, or with
	prominences greater than the specified minprom, will be considered. If
	prommode is 'absolute' or 'abs', minprom is an absolute cutoff. If
	prommode is 'relative' or 'rel', minprom is interpreted as a fraction
	of the highest prominence.
	'''
	try:
		absprom = {'abs': True, 'absolute': True,
				'rel': False, 'relative': False}[prommode.lower()]
	except KeyError: raise ValueError('Invalid argument for prommode')

	from .util import alternator

	# Prepare output extrema
	maxtab, mintab = [], []

	if (minwidth and minwidth < 0) or (minprom and minprom < 0):
		raise ValueError('Minimum prominence and width must be nonnegative')

	mn, mx = float('inf'), float('-inf')
	mnpos, mxpos = float('nan'), float('nan')

	lookformax = True

	# Alternately count all maxima and minima in the signal
	# Note: mintab[i] is always first minimum AFTER maxtab[i]
	for i, v in enumerate(vec):
		if v > mx: mx, mxpos = v, i
		if v < mn: mn, mnpos = v, i

		if lookformax and v < mx:
			maxtab.append((mxpos, mx))
			mn, mnpos = v, i
			lookformax = False
		elif not lookformax and v > mn:
			mintab.append((mnpos, mn))
			mx, mxpos = v, i
			lookformax = True

	# Build a list mapping each maximum to a key col
	keycols = [None] * len(maxtab)

	def bestcol(idx, lcol, rcol):
		'''
		Return the higher of lcol and rcol, or the col nearer to idx
		if both have the same height.

		If one of lcol or rcol is not a 2-tuple, the other is returned.
		'''
		try: li, lv = lcol
		except TypeError: return rcol

		try: ri, rv = rcol
		except TypeError: return lcol

		if lv > rv: return lcol
		if lv < rv: return rcol

		if abs(li - idx) < abs(ri - idx): return lcol
		else: return rcol


	# Walk to the right from each maximum in turn
	for i, (iidx, ival) in enumerate(maxtab):
		# Reset the col tracker
		col = None
		# Interleave extrema to right of start, starting with next minimum
		extrema = alternator(mintab[i:], maxtab[i+1:])
		# The first minimum has index 2 * i + 1 in the merged extrema list
		for j, (jidx, jval) in enumerate(extrema, 2 * i + 1):
			if j % 2:
				# Update col tracker if a lower col is encountered
				try:
					if jval <= col[1]: col = (jidx, jval)
				except TypeError: col = (jidx, jval)
				# Skip to next maximum for further processing
				continue
			# Analyze subsequent maxima
			if ival > jval:
				# Set candidate key col for lower peaks to right
				jh = j / 2
				keycols[jh] = bestcol(jidx, keycols[jh], col)
			else:
				# Final key col for left peak with higher peak at right
				keycols[i] = bestcol(iidx, keycols[i], col)
				# No need to walk past a larger peak
				break

	# Determine the overall minimum value
	minval = min(ex[1] for ex in mintab)

	if minprom and not absprom:
		# Reinterpret minprom if relative mode was specified
		minprom *= (max(ex[1] for ex in maxtab) - minval)

	if minwidth and minprom:
		# Assess both prominence and width for cuttof
		def goodpeak(width, prom):
			return (width >= minwidth) and (prom >= minprom)
	elif minwidth:
		# Assess only width for cutoff
		def goodpeak(width, prom):
			return width >= minwidth
	elif minprom:
		# Assess only prominence for cutoff
		def goodpeak(width, prom):
			return prom >= minprom
	else:
		# There is no cutoff
		def goodpeak(width, prom):
			return True


	# Build the peak list
	peaks = []
	for (ei, ev), kcol in izip(maxtab, keycols):
		try:
			# Try to unpack the key col
			ki, kv = kcol
			width = abs(ei - ki)
			prom = ev - kv
		except TypeError:
			# If the key col is "None", this is a dominant peak
			# The width stretches all the way to the end of the signal
			width = max(ei, len(vec) - ei)
			prom = ev - minval
		# Only record "good" peaks
		if goodpeak(width, prom):
			peaks.append((ei, ev, width, prom))

	return peaks


def shifter(sig, delays, s=None, axes=None):
	'''
	Shift a multidimensional signal sig by a number of (possibly
	fractional) sample units in each dimension specified as entries in the
	delays sequence. The shift is done using FFTs and the arguments s and
	axes take the same meaning as in numpy.fft.fftn.

	The length of delays must be compatible with the length of axes as
	specified or inferred.
	'''
	# Ensure that sig is a numpy.ndarray
	sig = asarray(sig)
	ndim = len(sig.shape)

	# Set default values for axes and s if necessary
	if axes is None:
		if s is not None: axes = range(ndim - len(s), ndim)
		else: axes = range(ndim)

	if s is None: s = tuple(sig.shape[a] for a in axes)

	# Check that all arguments agree
	if len(s) != len(axes):
		raise ValueError('FFT shape array and axes list must have same dimensionality')
	if len(s) != len(delays):
		raise ValueError('Delay list and axes list must have same dimensionality')

	# Take the forward transform for spectral shifting
	csig = fft.fftn(sig, s, axes)

	# Loop through the axes, shifting each one in turn
	for d, n, a in zip(delays, s, axes):
		# Build the FFT frequency indices
		dk = 2. * math.pi / n
		kidx = numpy.arange(n)
		k = dk * (kidx >= n / 2.).choose(kidx, kidx - n)
		# Build the shifter and the axis slicer for broadcasting
		sh = numpy.exp(-1j * k * d)
		slic = [numpy.newaxis] * ndim
		slic[a] = slice(None)
		# Multiply the shift
		csig *= sh[slic]

	# Perform the inverse transform and cast to the input type
	rsig = fft.ifftn(csig, axes=axes)
	if not numpy.issubdtype(sig.dtype, numpy.complexfloating):
		rsig = rsig.real
	return rsig.astype(sig.dtype)


def bandwidth(sigft, df=1, level=0.5, r2c=False):
	'''
	Return as (bw, fc) the bandwidth bw and center frequency fc of a signal
	whose DFT is given in sigft. The frequency bin width is df.

	The DFT is searched in both directions from the positive frequency bin
	with peak amplitude until the signal falls below the specified level.
	Linear interpolation pinpoints the level crossing between bins. The
	bandwidth is the difference between the high and low crossings,
	multiplied by df. Only the level crossing nearest the peak is
	identified in each direction.

	The center frequency is the average of the high and low crossing
	frequencies.

	If r2c is True, the DFT is assumed to contain only positive
	frequencies. Otherwise, the DFT should contain positive and negative
	frequencies in standard FFT order.
	'''
	sigamps = numpy.abs(sigft)
	if not r2c:
		# Strip the negative frequencies from the C2C DFT
		sigamps = sigamps[:len(sigamps)/2]

	# Find the peak positive frequency
	peakidx = numpy.argmax(sigamps)
	# Now scale the amplitudes
	sigamps /= sigamps[peakidx]


	# Search low frequencies for the level crossing
	flo = peakidx + 1
	for i, s in enumerate(reversed(sigamps[:peakidx])):
		if s < level:
			flo = peakidx - i
			break
	# Search high frequencies for the level crossing
	fhi = peakidx - 1
	for i, s in enumerate(sigamps[peakidx+1:]):
		if s < level:
			fhi = peakidx + i
			break

	# Ensure that a crossing level was identified
	if sigamps[flo - 1] > level:
		raise ValueError('Low-frequency level crossing not identified')
	if sigamps[fhi + 1] > level:
		raise ValueError('High-frequency level crossing not identified')

	# Now convert the indices to interpolated frequencies
	# The indices point to the furthest sample exceeding the level
	mlo = (sigamps[flo] - sigamps[flo - 1])
	mhi = (sigamps[fhi + 1] - sigamps[fhi])

	flo = (level - sigamps[flo - 1]) / float(mlo) + flo - 1
	fhi = (level - sigamps[fhi]) / float(mhi) + fhi

	bw = (fhi - flo) * df
	fc = 0.5 * (fhi + flo) * df
	return bw, fc


def psnr (x, y):
	'''
	The peak SNR, in dB, of a matrix x relative to the matrix y.
	This assumes x = y + N, where N is noise and y is signal.
	'''
	# Compute the average per-pixel squared error
	err = numpy.sum (numpy.abs(x - y)**2) / float(prod(x.shape))
	# Compute the square of the maximum signal value
	maxval = numpy.max(numpy.abs(y))**2

	# Compute the peak SNR in dB
	return 10. * math.log10(maxval / err)
