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
