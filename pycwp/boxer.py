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
		if len(start) != 3 or len(end) != 3:
			raise TypeError('Start and end points must be sequences of length 3')
		# Store an immutable, float copy of the start and end points
		self._start = tuple(float(s) for s in start)
		self._end = tuple(float(e) for e in end)

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
		return math.sqrt(sum((e - s)**2 for s, e in zip(self.start, self.end)))

	@lazy_property
	def direction(self):
		# Compute the normalized direction lazily
		return tuple(float(e - s) / self.length for s, e in zip(self.start, self.end))

	@lazy_property
	def invdirection(self): 
		return tuple(1. / d if d != 0. else float('inf') for d in self.direction)

	@lazy_property
	def midpoint(self): 
		return tuple(0.5 * (s + e) for s, e in zip(self.start, self.end))

	@lazy_property
	def majorAxis(self):
		return max(enumerate(self.direction), key=op.itemgetter(1))[0]

	def pointAtLength(self, t):
		'''
		For a given signed length t, return the point on the line
		through this segment which is a distance t from the start.
		'''
		return tuple(s + t * d for s, d in zip(self.start, self.direction))

	def lengthToPlane(self, c, axis):
		'''
		Return the signed distance along the segment from the start to
		the plane defined by a constant value c in the specified axis.

		If the segment and plane are parallel, the result will be
		signed infinity if the plane does not contain the segment.
		If the plane contains the segment, the length will be 0.
		'''
		start = self.start
		# Catch equality to zero to avoid NaN in parallel cases
		if c - start[axis] == 0.: return 0.
		return (c - start[axis]) * self.invdirection[axis]

	def projection(self, point):
		'''
		Project the three-dimensional point onto the ray defined by the
		start and direction of this segment.
		'''
		if len(point) != 3:
			raise TypeError('Point to project must be sequence of length 3')
		# Compute the differential point
		pdx = tuple(p - s for p, s in zip(point, self.start))
		# Take the inner product
		return sum(p * d for p, d in zip(pdx, self.direction))

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
		if len(lo) != 3 or len(hi) != 3:
			raise TypeError('Corners must be sequences of length 3')
		if any(l > h for l, h in zip(lo, hi)):
			raise ValueError('All coordinates of hi must not be less than coordinates of lo')

		# Store an immutable, float copy of the corners
		self._lo = tuple(float(l) for l in lo)
		self._hi = tuple(float(h) for h in hi)

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
		if len(c) != 3:
			raise TypeError('Cell numbers must be a 3-element sequence')
		if any(cl < 1 for cl in c):
			raise ValueError('Cell counts in all dimensions must be at least 1')
		# Set the cell count
		self._ncell = tuple(int(cl) for cl in c)
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
			self._cell = tuple(l / float(n) for l, n in zip(self.length, self.ncell))
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
		ci, cj, ck = [int(cv) for cv in c]
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
		for idx in itertools.product(*[xrange(nc) for nc in self.ncell]):
			box = self.getCell(idx)
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

		# Check for intersections within segment
		#return tmax >= max(0, tmin) and tmin <= segment.length

	def raymarcher(self, segment):
		'''
		Marches along the given 3-D line segment (like Segment3D) to
		identify all cells in the grid (defined by the ncell property)
		that intersect the segment. Returns a list of tuples
		representing the grid indices of each cell.
		'''
		# March along the major axis
		axis = segment.majorAxis
		# Record the transverse axes as tx, ty
		tx, ty = (axis + 1) % 3, (axis + 2) % 3

		# Pull grid parameters for efficiency
		ncell, cell, lo = self.ncell, self.cell, self.lo

		# Try to grab the intersection lengths, if they exist
		try: tmin, tmax = self.intersection(segment)
		except TypeError: return None

		# Find the intersection points
		pmin = segment.pointAtLength(max(0., tmin))
		pmax = segment.pointAtLength(min(segment.length, tmax))

		# Find the minimum and maximum slabs; ensure proper ordering
		slabmin = int((pmin[axis] - lo[axis]) / cell[axis])
		slabmax = int((pmax[axis] - lo[axis]) / cell[axis])
		if slabmin > slabmax: slabmin, slabmax = slabmax, slabmin
		# Ensure the slabs don't run off the end
		if slabmin < 0: slabmin = 0
		if slabmax >= ncell[axis]: slabmax = ncell[axis] - 1

		# Compute the starting position of the first slab
		dslab = cell[axis]
		esc = lo[axis] + dslab * slabmin

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
			# If the transverse coordinates don't change,
			# the entry cell is the only intersecting cell
			if entry[tx] == exit[tx] and entry[ty] == exit[ty]:
				intersections.append(entry)
				continue

			# If the transverse coordinates do change, check
			# all cells in the neighborhood of the change
			ranges = [(e,) for e in entry]
			for t in (tx, ty):
				# Determine whether a range is necessary
				# along each transverse axis
				if entry[t] != exit[t]:
					mn = max(0, min(entry[t], exit[t]))
					mx = min(ncell[t], max(entry[t], exit[t]) + 1)
					ranges[t] = xrange(mn, mx)

			for cell in itertools.product(*ranges):
				if self.getCell(cell).intersection(segment) is not None:
					intersections.append(cell)
		return intersections

	def raytracer(self, segments):
		'''
		Given a list of 3-D line segments, determine which of the cells
		in this box (as defined by the ncell property) intersect with
		the segments. Returns a list of tuples for each segment, each
		of the form (i, j, k), representing the grid indices of cells
		that intersect with that segment.
		'''
		# Find the number of levels necessary to enclose the whole box
		nmax = max(self.ncell)
		nc, y, nlev = nmax, 1, 0
		while nc > 1:
			nc >>= 1
			y <<= 1
			nlev += 1
		if y < nmax:
			y <<= 1
			nlev += 1

		# Add an extended box, with an oct-tree grid that aligns
		# with the desired box grid, to the investigation queue
		lo = self.lo
		hi = tuple(s + y * sl for s, sl in zip(lo, self.cell))
		boxqueue = [Box3D(lo, hi)]
		
		# Build a list of intersecting boxes for each segment
		results = [[] for i in range(len(segments))]

		for lev in range(nlev):
			# Intersection tests short-circuit after the first hit
			# on all but the finest level of computation
			shortCircuit = lev < nlev - 1
			# Create a new box queue for children in this level
			nextqueue = []
			# Iterate over all boxes in the current queue
			for ibox in boxqueue:
				# Loop over all child boxes in an oct-tree decomposition
				ibox.ncell = (2, 2, 2)
				for cbox in ibox.allCells():
					# Skip the child if it is outside this box
					if not self.overlaps(cbox): continue
	
					# When short-circuiting, add hit boxes to
					# next-level queue; at the finest level, add
					# cell index for each hit box to result list
					for i, segment in enumerate(segments):
						if cbox.intersection(segment) is not None:
							if shortCircuit:
								nextqueue.append(cbox)
								break
							else:
								mx, my, mz = cbox.midpoint
								ci = self.cart2cell(mx, my, mz, False)
								results[i].append(ci)
	
			# Populate the box queue with the intersecting children
			boxqueue = nextqueue

		return tuple(results)
