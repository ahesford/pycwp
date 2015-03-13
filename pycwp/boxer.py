'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

import sys, math, itertools
import operator as op

from .util import lazy_property

class Segment3D(object):
	'''
	A representation of a 3-D line segment.
	'''
	def __init__(self, start, end):
		'''
		Initialize a 3-D line segment that starts and ends at the
		indicated points.
		'''
		# Unpack the tuples to confirm dimensionality
		try:
			lx, ly, lz = start
			hx, hy, hz = end
		except ValueError:
			raise ValueError('Start and end points must be sequences of length 3')
		# Store an immutable, float copy of the start and end points
		self._start = float(lx), float(ly), float(lz)
		self._end = float(hx), float(hy), float(hz)

	@property
	def start(self):
		'''The starting point of the segment'''
		return self._start
	@property
	def end(self):
		'''The ending point of the segment'''
		return self._end

	@lazy_property
	def length(self): 
		# Compute the line length lazily
		sx, sy, sz = self.start
		ex, ey, ez = self.end
		return math.sqrt((ex - sx)**2 + (ey - sy)**2 + (ez - sz)**2)

	@lazy_property
	def direction(self):
		# Compute the normalized direction lazily
		sx, sy, sz = self.start
		ex, ey, ez = self.end
		length = self.length
		return (ex - sx) / length, (ey - sy) / length, (ez - sz) / length

	@lazy_property
	def invdirection(self): 
		return tuple(1. / d if d != 0. else float('inf') for d in self.direction)

	@lazy_property
	def midpoint(self): 
		sx, sy, sz = self.start
		ex, ey, ez = self.end
		return 0.5 * (sx + ex), 0.5 * (sy + ey), 0.5 * (sz + ez)

	@lazy_property
	def majorAxis(self):
		return max(enumerate(self.direction), key=op.itemgetter(1))[0]

	def pointAtLength(self, t):
		'''
		For a given signed length t, return the point on the line
		through this segment which is a distance t from the start.
		'''
		sx, sy, sz = self.start
		dx, dy, dz = self.direction

		return sx + t * dx, sy + t * dy, sz + t * dz

	def lengthToPlane(self, c, axis):
		'''
		Return the signed distance along the segment from the start to
		the plane defined by a constant value c in the specified axis.

		If the segment and plane are parallel, the result will be
		signed infinity if the plane does not contain the segment.
		If the plane contains the segment, the length will be 0.
		'''
		dx = c - self.start[axis]
		# Catch equality to zero to avoid NaN in parallel cases
		return 0 if (dx == 0.) else (dx * self.invdirection[axis])

	def projection(self, point):
		'''
		Project the three-dimensional point onto the ray defined by the
		start and direction of this segment.
		'''
		# Unpack the tuple to confirm dimensionality
		try: px, py, pz = point
		except ValueError: raise ValueError('Point must be a sequence of length 3')

		sx, sy, sz = self.start
		dx, dy, dz = self.direction

		return dx * (px - sx) + dy * (py - sy) + dz * (pz - sz)

	def __repr__(self):
		return self.__class__.__name__ + '(' + str(self.start) + ', ' + str(self.end) + ')'


class Box3D(object):
	'''
	A representation of an axis-aligned 3-D bounding box.
	'''
	def __init__(self, lo, hi):
		'''
		Initialize a 3-D box with extreme corners lo and hi.
		'''
		# Unpack tuples to confirm dimensionality
		try:
			lx, ly, lz = lo
			hx, hy, hz = hi
		except ValueError:
			raise ValueError('Corners must be sequences of length 3')

		if hx < lx or hy < ly or hz < lz:
			raise ValueError('All coordinates of hi must not be less than coordinates of lo')

		# Store an immutable, float copy of the corners
		self._lo = float(lx), float(ly), float(lz)
		self._hi = float(hx), float(hy), float(hz)

	@property
	def lo(self):
		'''The lower corner of the box'''
		return self._lo
	@property
	def hi(self):
		'''The upper corner of the box'''
		return self._hi

	@property
	def ncell(self):
		'''The number of cells, per dimension, which subdivide the box'''
		try: return self._ncell
		except AttributeError:
			# Set a default cell count
			self._ncell = (1, 1, 1)
			# Clear existing cell-length property to recompute as needed
			del self.cell
			return self._ncell

	@ncell.setter
	def ncell(self, c):
		# Unpack tuple to confirm dimensionality
		try: cx, cy, cz = c
		except ValueError:
			raise ValueError('Grid dimensions must be a 3-element sequence')

		if cx < 1 or cy < 1 or cz < 1:
			raise ValueError('Grid dimensions must each be at least 1')
		# Set the cell count
		self._ncell = int(cx), int(cy), int(cz)
		# Clear the cell-length property so it will be recomputed on demand
		del self.cell

	@ncell.deleter
	def ncell(self): 
		try: del self._ncell
		except AttributeError: pass
		del self.cell

	@property
	def cell(self):
		'''The lengths of each cell in the grid defined by ncell'''
		try: return self._cell
		except AttributeError:
			lx, ly, lz = self.length
			nx, ny, nz = self.ncell
			self._cell = lx / float(nx), ly / float(ny), lz / float(nz)
			return self._cell

	@cell.deleter
	def cell(self): 
		try: del self._cell
		except AttributeError: pass

	@lazy_property
	def midpoint(self): 
		return tuple(0.5 * (l + h) for l, h in zip(self.lo, self.hi))

	@lazy_property
	def length(self):
		return tuple(h - l for l, h in zip(self.lo, self.hi))

	def __repr__(self):
		return self.__class__.__name__ + '(' + str(self.lo) + ', ' + str(self.hi) + ')'

	def cart2cell(self, x, y, z, frac=True):
		'''
		Convert the 3-D Cartesian coordinates (x, y, z) into grid
		coordinates defined by the box bounds and ncell property. If
		frac is True, coordinates can have a fractional part to
		indicate relative positions in the cell. Otherwise, integer
		indices are returned.
		'''
		lx, ly, lz = self.lo
		cx, cy, cz = self.cell

		px = (x - lx) / cx
		py = (y - ly) / cy
		pz = (z - lz) / cz

		if not frac:
			px = int(px)
			py = int(py)
			pz = int(pz)

		return px, py, pz

	def cell2cart(self, i, j, k):
		'''
		Convert the (possibly fractional) 3-D cell-index coordinates
		(i, j, k), defined by the box bounds and ncell property, into
		Cartesian coordinates.
		'''
		lx, ly, lz = self.lo
		cx, cy, cz = self.cell

		px = i * cx + lx
		py = j * cy + ly
		pz = k * cz + lz

		return px, py, pz

	def getCell(self, c):
		'''
		Return a Box3D representing the cell that contains 3-D index c
		based on the grid defined by the ncell property.

		If c does not contain integer types, the types will be
		truncated. Cells outside the bounds of this box are allowed.
		'''
		try:
			ci, cj, ck = c
		except ValueError:
			raise ValueError('Cell index must be a 3-element sequence')
		ci, cj, ck = int(ci), int(cj), int(ck)
		lo = self.cell2cart(ci, cj, ck)
		hi = self.cell2cart(ci + 1, cj + 1, ck + 1)
		return Box3D(lo, hi)

	def allCells(self, enum=False):
		'''
		Return a generator that produces every cell in the grid defined
		by the ncell property. Generation is done in row-major order.

		If enum is True, return a tuple (idx, box), where idx is the
		three-dimensional index of the cell.
		'''
		ni, nj, nk = self.ncell
		for i in xrange(ni):
			for j in xrange(nj):
				for k in xrange(nk):
					box = self.getCell((i, j, k))
					if not enum: yield box
					else: yield (idx, box)

	def overlaps(self, b):
		'''
		Returns True iff the Box3D b overlaps with this box.
		'''
		al, ah = self.lo, self.hi
		bl, bh = b.lo, b.hi

		# Self is completely behind b
		if ah[0] < bl[0]: return False
		# Self is completely in front of b
		if al[0] > bh[0]: return False
		# Self is completely left of b
		if ah[1] < bl[1]: return False
		# Self is completely right of b
		if al[1] > bh[1]: return False
		# Self is completely below b
		if ah[2] < bl[2]: return False
		# Self is completely above b
		if al[2] > bh[2]: return False
		
		return True

	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in the box.
		'''
		x, y, z = p
		(lx, ly, lz), (hx, hy, hz) = self.lo, self.hi
		return ((x >= lx) and (x <= hx)
				and (y >= ly) and (y <= hy)
				and (z >= lz) and (z <= hz))

	def intersection(self, segment):
		'''
		Returns the lengths tmin and tmax along the given line segment
		(like Segment3D) at which the segment enters and exits the box.
		If the box does not intersect the segment, returns None.

		If the segment starts within the box, tmin will be negative. If
		the segment ends within the box, tmax will exceed the segment
		length.
		'''
		sx, sy, sz = segment.start
		idx, idy, idz = segment.invdirection
		lx, ly, lz = self.lo
		hx, hy, hz = self.hi

		# Check, in turn, intersections with the x, y and z slabs
		tmin = (lx - sx) * idx
		tmax = (hx - sx) * idx
		if tmax < tmin: tmin, tmax = tmax, tmin
		# Check the y-slab
		ty1 = (ly - sy) * idy
		ty2 = (hy - sy) * idy
		if ty2 < ty1: ty1, ty2 = ty2, ty1
		if ty2 < tmax: tmax = ty2
		if ty1 > tmin: tmin = ty1
		# Check the z-slab
		tz1 = (lz - sz) * idz
		tz2 = (hz - sz) * idz
		if tz2 < tz1: tz1, tz2 = tz2, tz1
		if tz2 < tmax: tmax = tz2
		if tz1 > tmin: tmin = tz1

		if tmax < max(0, tmin) or tmin > segment.length: return None
		return tmin, tmax

	def raymarcher(self, segment):
		'''
		Marches along the given 3-D line segment (like Segment3D) to
		identify all cells in the grid (defined by the ncell property)
		that intersect the segment. Returns a list of tuples of the
		form (i, j, k, tmin, tmax), where (i, j, k) is a grid index of
		a cell, and (tmin, tmax) are the lengths along the segment of
		the entry and exit points, respectively, through the cell.

		If the segment begins or ends in a cell, tmin or tmax may fall
		outside the range [0, segment.length].
		'''
		# March along the major axis
		axis = segment.majorAxis
		# Record the transverse axes as tx, ty
		tx, ty = (axis + 1) % 3, (axis + 2) % 3

		# Pull grid parameters for efficiency
		ncell = self.ncell
		lo = self.lo[axis]
		dslab = self.cell[axis]

		nax, ntx, nty = ncell[axis], ncell[tx], ncell[ty]

		# Try to grab the intersection lengths, if they exist
		try: tmin, tmax = self.intersection(segment)
		except TypeError: return []

		# Find the intersection points
		pmin = segment.pointAtLength(max(0., tmin))
		pmax = segment.pointAtLength(min(segment.length, tmax))

		# Find the minimum and maximum slabs; ensure proper ordering
		slabmin = int((pmin[axis] - lo) / dslab)
		slabmax = int((pmax[axis] - lo) / dslab)
		if slabmin > slabmax: slabmin, slabmax = slabmax, slabmin
		# Ensure the slabs don't run off the end
		if slabmin < 0: slabmin = 0
		if slabmax >= nax: slabmax = nax - 1

		# Compute the starting position of the first slab
		esc = lo + dslab * slabmin

		# Build a list of slab indices and segment entry points
		# Add an extra slab to capture exit from the final slab
		slabs = []
		for slab in range(slabmin, slabmax + 2):
			# Figure the segment length at the edge of the slab
			t = segment.lengthToPlane(esc, axis)
			# Shift the edge coordinate to the next slab
			esc += dslab
			# Find the coordinates of entry cell
			px, py, pz = segment.pointAtLength(t)
			idx = self.cart2cell(px, py, pz, False)
			# Add the cell coordinates to the list
			# Override axial index to correct rounding errors
			slabs.append(idx[:axis] + (slab,) + idx[axis+1:])

		intersections = []

		# Now enumerate all intersecting cells
		for entry, exit in itertools.izip(slabs, slabs[1:]):
			ranges = [(e,) for e in entry]
			# Check all cells in range spanned by transverse indices
			enx, eny = entry[tx], entry[ty]
			exx, exy = exit[tx], exit[ty]
			if enx != exx:
				mn, mx = (enx, exx) if (enx < exx) else (exx, enx)
				ranges[tx] = xrange(max(0, mn), min(ntx, mx + 1))
			if eny != exy:
				mn, mx = (eny, exy) if (eny < exy) else (exy, eny)
				ranges[ty] = xrange(max(0, mn), min(ntx, mx + 1))

			for i in ranges[0]:
				for j in ranges[1]:
					for k in ranges[2]:
						t = self.getCell((i, j, k)).intersection(segment)
						if t is not None:
							intersections.append((i, j, k, t[0], t[1]))
		return intersections
