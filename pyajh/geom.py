import math

def sphsegvol(r, c, lh, ll):
	'''
	Compute the volume of a sphere segment with radius r, center elevation
	c, upper bound elevation lh, and lower bound elevation ll.
	'''

	# Compute the radii of the upper and lower bases
	asq = max(0., r**2 - (lh - c)**2)
	bsq = max(0., r**2 - (ll - c)**2)

	# Compute the elevation of the upper and lower bases
	pmax = min(r, lh - c)
	pmin = max(-r, ll - c)

	# Compute the segment height
	h = max(0., pmax - pmin)

	# Return the segment volume
	return (math.pi / 6.) * h * (h**2 + 3 * (asq + bsq))

def cyleqvrad(v, h):
	'''
	Compute the radius of a cylinder with volume v and height h.
	'''

	return math.sqrt(v / math.pi / h)

def sph2cart (r, t, p):
	'''
	Mimics the sph2cart function of Matlab, converting spherical
	coordinates (r, t, p) to Cartesian coordinates (x, y, z). The
	variables t and p are the polar and azimuthal angles, respectively.
	'''
	st = numpy.sin(t)
	return r * st * numpy.cos(p), r * st * numpy.sin(p), r * numpy.cos(t)

def cart2sph (x, y, z):
	'''
	Mimics the cart2sph function of Matlab, converting Cartesian
	coordinates (x, y, z) to spherical coordinates (r, t, p). The
	variables t and p are the polar and azimuthal angles, respectively.
	'''
	r = numpy.sqrt(x**2 + y**2 + z**2)
	t = numpy.arccos (z / r)
	p = numpy.arctan2 (y, x)
	return r, t, p

def pol2cart (r, t):
	'''
	Mimics the pol2cart function of Matlab, converting polar coordinates
	(r, t) to Cartesian coordinates (x, y).
	'''
	return r * numpy.cos(t), r * numpy.sin(t)

def cart2pol (x, y):
	'''
	Mimics the cart2pol function of Matlab, converting Cartesian
	coordinates (x, y) to polar coordinates (r, t).
	'''
	r = numpy.sqrt (x**2 + y**2)
	t = numpy.arctan2 (y, x)
	return r, t
