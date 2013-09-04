import math, numpy as np


def rotate3d(pt, theta, phi, inverse=False):
	'''
	Rotate a 3-D point pt (represented as a sequence x, y, z) by first
	rotating by an azimuthal angle phi about the z axis, then rotating by a
	polar angle theta about the new y axis.

	If inverse is True, first rotate the polar angle about the y axis and
	then rotate the azimuthal angle about the new z axis.
	'''
	cphi = math.cos(phi)
	sphi = math.sin(phi)
	ctheta = math.cos(theta)
	stheta = math.sin(theta)

	x, y, z = pt

	if not inverse:
		# First perform the azimuthal rotation
		nx = x * cphi - y * sphi
		ny = x * sphi + y * cphi
		# Next perform the polar rotation
		nz = z * ctheta - nx * stheta
		nx = z * stheta + nx * ctheta
	else:
		# First perform the polar rotation
		nz = z * ctheta - x * stheta
		nx = z * stheta + x * ctheta
		# Next perform the azimuthal rotation
		ny = nx * sphi + y * cphi
		nx = nx * cphi - y * sphi

	return nx, ny, nz


def gridcoords(i, nc, dc, dim=3):
	'''
	Return the Cartesian coordinates of the center of cell i in a grid of
	dimension dim (1, 2, or 3). The shape of the grid nc is a list of
	either dimension dim or, for isotropic grids, a scalar value. Likewise,
	dc may be of dimension dim to specify rectangular elements, or a scalar
	to specify cubic elements. The last coordinate is most rapidly varying.
	'''

	# Ensure proper bounds on dimension
	if dim < 1 or dim > 3:
		raise ValueError("Dimension must be 1, 2, or 3")

	# Check for argument sanity
	if len(nc) != dim and len(nc) != 1:
		raise ValueError("Grid size must be a list of dimension 1 or %d" % dim)
	if len(dc) != dim and len(dc) != 1:
		raise ValueError("Cell size must be a list of dimension 1 or %d" % dim)

	# Expand one-dimensional arrays
	if len(nc) == 1: nc = [nc[0]] * dim
	if len(dc) == 1: dc = [dc[0]] * dim

	if dim == 1:
		coords = np.array(i % nc[0])
	elif dim == 2:
		coords = np.array((i / nc[1], i % nc[1]))
	elif dim == 3:
		coords = np.array((i / (nc[1] * nc[2]), (i / nc[1]) % nc[2], i % nc[2]))

	# Return the coordinates
	return  ((2. * coords + 1. - np.array(nc)) * np.array(dc) * 0.5).tolist()


def sampcoords(i, nt, np, th = None):
	'''
	Compute the Cartesian position of sample i in a spherical grid
	involving nt theta samples and np phi samples. The theta sample
	positions can be provided, or they will be computed as uniform
	samples in the interval [0, pi] Gauss-Legendre polynomials. Phi
	samples are uniform in the interval [0, 2*pi).
	'''

	# The first sample is always the south pole
	if i == 0: return (0., 0., -1.)

	# Compute uniform theta samples, if none are provided
	if th is None:
		th = [math.pi * i / float(nt - 1) for i in xrange(nt)]
		th.reverse()

	# Compute the indices
	pidx = (i - 1) % np
	tidx = 1 + (i - 1) / np

	# The last sample is always the north pole
	if tidx == nt - 1: return (0., 0., 1.)

	# Compute the angular position
	phi = 2 * math.pi * pidx / float(np)
	theta = th[tidx]

	return (math.cos(phi) * math.sin(theta),
			math.sin(phi) * math.sin(theta), math.cos(theta))


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
	st = np.sin(t)
	return r * st * np.cos(p), r * st * np.sin(p), r * np.cos(t)

def cart2sph (x, y, z):
	'''
	Mimics the cart2sph function of Matlab, converting Cartesian
	coordinates (x, y, z) to spherical coordinates (r, t, p). The
	variables t and p are the polar and azimuthal angles, respectively.
	'''
	r = np.sqrt(x**2 + y**2 + z**2)
	t = np.arccos (z / r)
	p = np.arctan2 (y, x)
	return r, t, p

def pol2cart (r, t):
	'''
	Mimics the pol2cart function of Matlab, converting polar coordinates
	(r, t) to Cartesian coordinates (x, y).
	'''
	return r * np.cos(t), r * np.sin(t)

def cart2pol (x, y):
	'''
	Mimics the cart2pol function of Matlab, converting Cartesian
	coordinates (x, y) to polar coordinates (r, t).
	'''
	r = np.sqrt (x**2 + y**2)
	t = np.arctan2 (y, x)
	return r, t
