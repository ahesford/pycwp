'''
Classes to represent axis-aligned 3-D bounding boxes and 3-D line segments, and
to perform ray-tracing based on oct-tree decompositions or a linear marching
algorithm.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import sys, math, itertools

from itertools import izip, product as iproduct

from .util import lazy_property
from .cutil import dot, norm

def _infdiv(a, b):
	'''
	Return a / b with special handling of small values:

	1. If |b| < epsilon * |a| for machine epsilon, return signed infinity,
	2. Otherwise, if |a| < epsilon, return 0.
	'''
	aa, ab = abs(a), abs(b)
	eps = sys.float_info.epsilon

	if ab <= eps * aa:
		sa = 2 * int(a >= 0) - 1
		sb = 2 * int(b >= 0) - 1
		return sa * sb * float('inf')
	elif aa < eps:
		return 0.

	return a / b

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
		except (ValueError, TypeError):
			raise TypeError('Start and end points must be sequences of length 3')
		# Store an immutable, float copy of the start and end points
		self._start = float(lx), float(ly), float(lz)
		self._end = float(hx), float(hy), float(hz)

		# Compute the line length
		self._length = math.sqrt((hx - lx)**2 + (hy - ly)**2 + (hz - lz)**2)
		if abs(self._length) <= sys.float_info.epsilon:
			raise ValueError('Segment must have nonzero length')

	@property
	def start(self):
		'''The starting point of the segment'''
		return self._start

	@property
	def end(self):
		'''The ending point of the segment'''
		return self._end

	@property
	def length(self):
		'''
		The length of the line segment.
		'''
		return self._length

	@lazy_property
	def direction(self):
		# Compute the normalized direction lazily
		length = self.length
		return tuple((e - s) / length for s,e in izip(self.start, self.end))

	@lazy_property
	def midpoint(self):
		sx, sy, sz = self.start
		ex, ey, ez = self.end
		return 0.5 * (sx + ex), 0.5 * (sy + ey), 0.5 * (sz + ez)

	@lazy_property
	def majorAxis(self):
		return max(enumerate(self.direction), key=lambda x: abs(x[1]))[0]

	def cartesian(self, t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		sx, sy, sz = self.start
		dx, dy, dz = self.direction

		return sx + t * dx, sy + t * dy, sz + t * dz

	def bezier(self, p, project=False):
		'''
		For a given 3-D Cartesian point p, return the signed length t
		such that p, projected along the segment, is a distance t from
		its start.

		If project is False and the point is not on the line, None will
		be returned.
		'''
		dl = self.direction
		dp = tuple(v - w for v, w in izip(p, self.start))
		t = dot(dp, dl)

		if not project and norm(v - t * d for v, d in izip(dp, dl)) > sys.float_info.epsilon:
			return None

		return t

	def lengthToAxisPlane(self, c, axis):
		'''
		Return the signed distance along the segment from the start to
		the plane defined by a constant value c in the specified axis.

		If the segment and plane are parallel, the result will be
		signed infinity if the plane does not contain the segment.
		If the plane contains the segment, the length will be 0.
		'''
		dx = c - self.start[axis]
		return _infdiv(dx, self.direction[axis])

	def __repr__(self):
		return self.__class__.__name__ + '(' + repr(self.start) + ', ' + repr(self.end) + ')'


class Triangle3D(object):
	'''
	A representation of a triangle embedded in 3-D space.
	'''
	@staticmethod
	def _checkdims(p):
		'''
		Ensure that the argument p is a three-length sequence, or else
		raise a TypeError.
		'''
		try:
			if len(p) != 3: raise TypeError
		except TypeError:
			raise TypeError('Sequence must have length 3')

	def __init__(self, nodes):
		'''
		Initialize a triangle from nodes in the given sequence.
		'''
		self._checkdims(nodes)

		try:
			self.nodes = tuple((float(x), float(y), float(z)) for x, y, z in nodes)
		except (ValueError, TypeError):
			raise TypeError('Each element of nodes must be a sequence of 3 floats')

		def crosser(i):
			n0 = self.nodes[i]
			n1 = self.nodes[(i + 1) % 3]
			n2 = self.nodes[(i + 2) % 3]
			vx, vy, vz = [rv - lv for rv, lv in izip(n1, n0)]
			wx, wy, wz = [rv - lv for rv, lv in izip(n2, n0)]
			return vy * wz - vz * wy, vz * wx - vx * wz, vx * wy - vy * wx

		mag, nrm = max((norm(v), v)
				for i in xrange(3) for v in [crosser(i)])

		# Length of cross product of two sides is twice triangle area
		self._area = 0.5 * mag
		if abs(self._area) <= sys.float_info.epsilon:
			raise ValueError('Triangle must have nonzero area')

		# Scale the normal
		self._normal = tuple(v / mag for v in nrm)

	def __repr__(self):
		return self.__class__.__name__ + '(' + repr(self.nodes) + ')'

	@property
	def normal(self):
		'''
		The unit normal to the plane containing the triangle.
		'''
		return self._normal

	@property
	def area(self):
		'''
		The area of the triangle.
		'''
		return self._area

	@lazy_property
	def edges(self):
		'''
		Return a list of line segments that define the sides of the
		triangle, with segment i starting at self.node[i] and ending at
		self.node[(i + 1) % 3] for i in range(3).
		'''
		nodes = self.nodes
		return tuple(Segment3D(nodes[i], nodes[(i + 1) % 3]) for i in xrange(3))

	@lazy_property
	def qrbary(self):
		'''
		Returns, as (q, r), a representation of the thin QR
		factorization of the overdetermined matrix that relates
		barycentric coordinates l = [l[0], l[1]] to the coordinates of
		p - self.nodes[2] for some 3-D point p. The third barycentric
		coordinate l[2] = 1 - l[1] - l[0].

		The matrix Q is represented as q = (q[0], q[1]), where q[0] and
		q[1] are the two, three-dimensional, orthonormal column vectors
		spanning the range of the relational matrix.

		The matrix R is represented as r = (r[0], r[1], r[2]), such
		that R = [ r[0], r[1]; 0 r[2] ].

		The system will have no solution if the point p is not in the
		plane of the triangle.
		'''
		# Perform Gram-Schmidt orthogonalization to build Q and R
		nodes = self.nodes
		q0 = tuple(l - r for l, r in izip(nodes[0], nodes[2]))
		r = (norm(q0),)
		if r[0] <= sys.float_info.epsilon:
			raise ValueError('Barycentric coordinates cannot be found in degenerate triangles')
		q0 = tuple(v / r[0] for v in q0)

		q1 = tuple(l - r for l, r in izip(nodes[1], nodes[2]))
		r += (dot(q1, q0),)
		q1 = tuple(v - r[1] * w for v, w in izip(q1, q0))
		r += (norm(q1),)
		if r[2] <= sys.float_info.epsilon:
			raise ValueError('Barycentric coordinates cannot be found in degenerate triangles')
		q1 = tuple(v / r[2] for v in q1)

		return (q0, q1), r

	def barycentric(self, p, project=False):
		'''
		Convert the 3-D Cartesian point p into barycentric coordinates
		for this triangle. If project is True, the coordinates will be
		computed for the projection of the point onto the plane of the
		triangle. If project is False and the point is not in the plane
		of the triangle, None will be returned.
		'''
		self._checkdims(p)

		d = tuple(v - w for v, w in izip(p, self.nodes[2]))

		# Make sure the point is in the plane, if necessary
		if not project and abs(dot(d, self.normal)) > sys.float_info.epsilon:
			return None

		# Invert the orthogonal part of the QR system
		Q, R = self.qrbary
		q = tuple(dot(d, b) for b in Q)

		# Invert the triangular part of the QR system
		x1 = q[1] / R[2]
		x0 = (q[0] - R[1] * x1) / R[0]

		return (x0, x1, 1 - x0 - x1)

	def cartesian(self, p):
		'''
		For a point p in barycentric coordinates, return the
		corresponding Cartesian coordinates.
		'''
		self._checkdims(p)

		return tuple(dot(p, v) for v in izip(*self.nodes))

	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in this triangle.
		'''
		try:
			# Check the ranges of barycentric coordinates
			return all(0 <= v <= 1 for v in self.barycentric(p))
		except ValueError:
			# Barycentric conversion fails for out-of-plane points
			return False

	def overlaps(self, b):
		'''
		Returns True iff the Box3D b overlaps with this triangle.
		'''
		# If any node is in the box, the triangle overlaps
		for node in self.nodes:
			if b.overlaps(node): return True

		# Otherwise, if any segment intersects the box, the triangle overlaps
		for edge in self.edges:
			if b.intersection(edge): return True

		return False


	def intersection(self, seg):
		'''
		Return the intersection of the segment seg with this triangle,
		specified in units of length along the segment.

		If the segment and triangle do not intersect, this method
		returns None.

		If the segment and triangle intersect and are parallel, this
		method returns the lengths tmin and tmax that tightly bound the
		intersection along the segment.

		If the segment and triangle intersect but are not parallel, a
		single length t that specifies the point of intersection will
		be returned.
		'''
		# Extend the barycentric QR factorization to a
		# factorization for parametric line-plane intersection
		(q0, q1), (r0, r1, r2) = self.qrbary

		# The last column of and RHS of the intersection problem
		q2 = tuple(a - b for a, b in izip(seg.start, seg.end))
		y = tuple(a - b for a, b in izip(seg.start, self.nodes[2]))

		# Use modified Gram-Schmidt to find the last column
		r3 = dot(q0, q2)
		q2 = tuple(v - r3 * w for v, w in izip(q2, q0))
		r4 = dot(q1, q2)
		q2 = tuple(v - r4 * w for v, w in izip(q2, q1))
		r5 = norm(q2)

		if r5 <= sys.float_info.epsilon:
			# Line is parallel to facet, either in or out of plane
			# TODO: Check in-plane case
			return None

		q2 = tuple(v / r5 for v in q2)

		# Invert the QR factorization
		y = tuple(dot(y, b) for b in (q0, q1, q2))
		v = y[2] / r5
		u = (y[1] - r4 * v) / r2
		t = (y[0] - r2 * v - r1 * u) / r0

		# In this problem, length along the segment is v
		if all(0 <= l <= 1 for l in (v, u, t)) and 0 <= u + t <= 1: return v
		else: return None


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
		except (ValueError, TypeError):
			raise TypeError('Corners must be sequences of length 3')

		# Store an immutable, float copy of the corners
		self._lo = float(lx), float(ly), float(lz)
		self._hi = float(hx), float(hy), float(hz)

		eps = sys.float_info.epsilon
		if any(hv - lv <= eps for lv, hv in izip(self._lo, self._hi)):
			raise ValueError('Coordinates of hi must be larger than those of lo')

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
		return self.__class__.__name__ + '(' + repr(self.lo) + ', ' + repr(self.hi) + ')'

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

	def allIndices(self):
		'''
		Return a generator that produces every 3-D cell index within
		the grid defined by the ncell property in row-major order.
		'''
		ni, nj, nk = self.ncell
		return iproduct(xrange(ni), xrange(nj), xrange(nk))

	def allCells(self, enum=False):
		'''
		Return a generator that produces every cell in the grid defined
		by the ncell property. Generation is done in the same order as
		self.allIndices().

		If enum is True, return a tuple (idx, box), where idx is the
		three-dimensional index of the cell.
		'''
		for idx in self.allIndices():
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
		try:
			sx, sy, sz = segment.start
			seglen = segment.length
			dx, dy, dz = segment.direction
		except AttributeError:
			raise TypeError('Argument segment must behave like Segment3D')

		lx, ly, lz = self.lo
		hx, hy, hz = self.hi

		# Check, in turn, intersections with the x, y and z slabs
		tmin = _infdiv(lx - sx, dx)
		tmax = _infdiv(hx - sx, dx)
		if tmax < tmin: tmin, tmax = tmax, tmin
		# Check the y-slab
		ty1 = _infdiv(ly - sy, dy)
		ty2 = _infdiv(hy - sy, dy)
		if ty2 < ty1: ty1, ty2 = ty2, ty1
		if ty2 < tmax: tmax = ty2
		if ty1 > tmin: tmin = ty1
		# Check the z-slab
		tz1 = _infdiv(lz - sz, dz)
		tz2 = _infdiv(hz - sz, dz)
		if tz2 < tz1: tz1, tz2 = tz2, tz1
		if tz2 < tmax: tmax = tz2
		if tz1 > tmin: tmin = tz1

		if tmax < max(0, tmin) or tmin > seglen: return None
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
		pmin = segment.cartesian(max(0., tmin))
		pmax = segment.cartesian(min(segment.length, tmax))

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
			t = segment.lengthToAxisPlane(esc, axis)
			# Shift the edge coordinate to the next slab
			esc += dslab
			# Find the coordinates of entry cell
			px, py, pz = segment.cartesian(t)
			idx = self.cart2cell(px, py, pz, False)
			# Add the cell coordinates to the list
			# Override axial index to correct rounding errors
			slabs.append(idx[:axis] + (slab,) + idx[axis+1:])

		intersections = []

		# Now enumerate all intersecting cells
		for entry, exit in izip(slabs, slabs[1:]):
			ranges = [(e,) for e in entry]
			# Check all cells in range spanned by transverse indices
			enx, eny = entry[tx], entry[ty]
			exx, exy = exit[tx], exit[ty]
			if enx != exx:
				mn, mx = (enx, exx) if (enx < exx) else (exx, enx)
				ranges[tx] = xrange(max(0, mn), min(ntx, mx + 1))
			if eny != exy:
				mn, mx = (eny, exy) if (eny < exy) else (exy, eny)
				ranges[ty] = xrange(max(0, mn), min(nty, mx + 1))

			for i in ranges[0]:
				for j in ranges[1]:
					for k in ranges[2]:
						t = self.getCell((i, j, k)).intersection(segment)
						if t is not None:
							intersections.append((i, j, k, t[0], t[1]))
		return intersections


class Octree(object):
	'''
	An multilevel oct-tree decomposition of a three-dimensional space,
	wherein each level has a Box3D root defining its limits, and contains
	up to eight Octree children (each representing an octant of the root).

	Leafy children (those children of level 0) can be added to the tree
	recursively and need not be Box3D objects.
	'''
	def __init__(self, level, box):
		'''
		Construct an oct tree with a depth of level. The created tree
		has a rootbox property that is a Box3D object with lo and hi
		bounds copied from the provided Box3D box. For levels greater
		than 0, the rootbox is subdivided into (2, 2, 2) octant cells.

		Each octant cell will be the rootBox for one child, but
		children are created lazily when leaves are added with the
		addleaves method. When created, the children will be
		accumulated in the children property of this object, which maps
		octant indices (i, j, k), where each index can take the value 0
		or 1, to an Octree instance at (level - 1).
		'''
		self.level = int(level)
		if self.level != level:
			raise TypeError('Argument level must be an integer')
		if self.level < 0:
			raise ValueError('Argument level must be nonnegative')

		self.rootbox = Box3D(box.lo, box.hi)
		self.children = { }

		# No further action is necessary at level 0
		if self.level < 1: return

		# Subdivide the root into octants and assign children
		self.rootbox.ncell = (2, 2, 2)

	def prune(self):
		'''
		Recursively remove all children that, themselves, have no
		children. For level-0 trees, this is a no-op.
		'''
		if self.level < 1: return

		# Keep track of empty children
		nokids = set()
		for k, v in self.children.iteritems():
			# Prune the child to see if it is empty
			v.prune()
			if not v.children: nokids.add(k)

		for idx in nokids:
			try: del self.children[idx]
			except KeyError: pass

	def addleaves(self, leaves, predicate, multibox=False):
		'''
		Add leaves to the tree by populating the children property of
		level-0 trees in the hierarchy that match the given predicate.
		If any branches are missing from the tree, they will be created
		as necessary.

		The leaves should be a mapping from some arbitrary global
		identifier to an some arbitrary leaf object. The keys and
		values of the level-0 children maps will be the keys and
		values, respectively, of the provided leaves.

		The predicate should be a callable that takes positional
		arguments box and leaf and returns True if the specified Box3D
		box contains the object leaf. The predicate will be called for
		boxes at every level of the tree while drilling down.

		If multibox is True, all level-0 boxes that match the predicate
		for a given leaf will record the leaf as a child. Otherwise,
		only the first box to satisfy the predicate will own the box.
		The tree is walked depth-first, with children encountered in
		the order determined by the method Box3D.allIndices.

		If no box contains an entry in leaves, that entry will be
		silently ignored.

		This method returns True if any leaf was added to the tree,
		False otherwise.
		'''
		rbox = self.rootbox
		added = False

		for k, v in leaves.iteritems():
			# Check whether the leaf belongs in this tree
			if not predicate(rbox, v): continue

			if self.level < 1:
				# Just add the children at the finest level
				self.children[k] = v
				added = True
				continue

			# At higher levels, try to add the leaf to children
			kv = { k: v }
			for idx in rbox.allIndices():
				# Grab or create a child tree
				try:
					ctree = self.children[idx]
				except KeyError:
					cbox = rbox.getCell(idx)
					ctree = Octree(self.level - 1, cbox)

				# Add the leaf to the child tree, if possible
				if ctree.addleaves(kv, predicate, multibox):
					added = True
					# Make sure the child is recorded
					self.children.setdefault(idx, ctree)
					if not multibox: break

		return added

	def search(self, predicate, leafpred=None):
		'''
		Perform a depth-first search of the tree. The order in which
		children are followed is arbitrary.

		The callable predicate should take a single Box3D argument,
		which will be the root of some branch of the Octree, and return
		True to continue searching down the branch. If the predicate
		evaluates to False, searching terminates along that branch.

		The optional callable leafpred should take as its sole argument
		a leaf previously assigned to the tree using addleaves(). Any
		leaf of a level-0 box that satisfies the predicate (along with
		all of its parents) will be included in the search results if
		leafpred(leaf) is True and ignored if leafpred(leaf) is False.
		If leafpred is not specified, the default implementation always
		returns True.

		*** NOTE: The semantics of the queries imply that no leaf will
		be included in the results unless it satisfies leafpred and
		*ALL* of its ancestor boxes satisfy predicate.

		The return value is a dictionary composed of the keys and
		values of all matching leaves.
		'''
		# Match is empty if the predicate fails
		if not predicate(self.rootbox): return { }

		if self.level > 0:
			# Recursively check branches
			results = { }
			for ctree in self.children.itervalues():
				results.update(ctree.search(predicate, leafpred))
			return results

		# With no leaf predicate, all leaves match
		if not leafpred: return dict(self.children)

		# Otherwise, filter leaves by the leaf predicate
		return { k: v for k, v in self.children.iteritems() if leafpred(v) }
