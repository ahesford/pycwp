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
from .cutil import dot, norm, almosteq

def _checkdims(p, d=3):
	'''
	Raise a TypeError unless the argument p is a sequence of length d.
	'''
	try:
		if len(p) != d: raise TypeError
	except TypeError:
		raise TypeError('Sequence must have length %d', (d,))

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
	elif aa <= eps:
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
		_checkdims(start)
		_checkdims(end)

		# Store an immutable, float copy of the start and end points
		self._start = tuple(float(x) for x in start)
		self._end = tuple(float(x) for x in end)

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
		'''
		The length of the line segment.
		'''
		return norm(x - y for x, y in izip(self._start, self._end))

	@lazy_property
	def direction(self):
		'''
		The direction of the segment, normalized if the length is
		nonzero.

		In the special case that the length of the segment is
		(approximately) zero, the direction will be (0., 0., 0.).
		'''
		length = self.length
		if almosteq(length, 0.0): return (0., 0., 0.)
		return tuple((e - s) / length for s,e in izip(self.start, self.end))

	@lazy_property
	def midpoint(self):
		sx, sy, sz = self.start
		ex, ey, ez = self.end
		return 0.5 * (sx + ex), 0.5 * (sy + ey), 0.5 * (sz + ez)

	@lazy_property
	def majorAxis(self):
		return max(enumerate(self.direction), key=lambda x: abs(x[1]))[0]

	def bbox(self):
		'''
		A Box3D instance that bounds the segment.
		'''
		return Box3D(*zip(*((min(j), max(j))
			for j in izip(self.start, self.end))))

	def cartesian(self, t):
		'''
		For a given signed length t, return the Cartesian point on the
		line through this segment which is a distance t from the start.
		'''
		return tuple(x + t * d for x, d in izip(self.start, self.direction))

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
	def __init__(self, nodes):
		'''
		Initialize a triangle from nodes in the given sequence.
		'''
		_checkdims(nodes)

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
		if almosteq(self._area, 0.0):
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
		if almosteq(r[0], 0.0):
			raise ValueError('Barycentric coordinates cannot be found in degenerate triangles')
		q0 = tuple(v / r[0] for v in q0)

		q1 = tuple(l - r for l, r in izip(nodes[1], nodes[2]))
		r += (dot(q1, q0),)
		q1 = tuple(v - r[1] * w for v, w in izip(q1, q0))
		r += (norm(q1),)
		if almosteq(r[2], 0.0):
			raise ValueError('Barycentric coordinates cannot be found in degenerate triangles')
		q1 = tuple(v / r[2] for v in q1)

		return (q0, q1), r

	def edges(self):
		'''
		A generator that yields the three edges of the triangle as
		Segment3D instances. The start node of edge i is self.node[i]
		and the end node is self.node[(i+1) % 3].
		'''
		nodes = self.nodes
		for i in xrange(3):
			yield Segment3D(nodes[i], nodes[(i+1) % 3])

	def bbox(self):
		'''
		A Box3D instance that bounds the triangle.
		'''
		return Box3D(*zip(*((min(j), max(j)) for j in izip(*self.nodes))))

	def barycentric(self, p, project=False):
		'''
		Convert the 3-D Cartesian point p into barycentric coordinates
		for this triangle. If project is True, the coordinates will be
		computed for the projection of the point onto the plane of the
		triangle. If project is False and the point is not in the plane
		of the triangle, None will be returned.
		'''
		_checkdims(p)

		d = tuple(v - w for v, w in izip(p, self.nodes[2]))

		# Make sure the point is in the plane, if necessary
		if not (project or almosteq(dot(d, self.normal), 0.0)):
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
		_checkdims(p)

		return tuple(dot(p, v) for v in izip(*self.nodes))

	def contains(self, p):
		'''
		Returns True iff the 3-D point p is contained in this triangle.
		'''
		try:
			# Check the ranges of barycentric coordinates
			return all(0 <= v <= 1 for v in self.barycentric(p))
		except (ValueError, TypeError):
			# Barycentric conversion fails for out-of-plane points
			return False

	def overlaps(self, b):
		'''
		Returns True iff the Box3D b overlaps with this triangle.
		'''
		# Check edge intersections with box
		return any(b.intersection(edge) for edge in self.edges())

	def intersection(self, seg):
		'''
		Return the intersection of the segment seg with this triangle
		as (l, t, u, v), where l is the length along the segment seg
		and (t, u, v) are the barycentric coordinates in the triangle.

		If the segment and triangle are in the same plane, this method
		raises a NotImplementedError without checking for intersection.

		Otherwise, the method returns the length t along the segment
		that defines the point of intersection. If the segment and
		triangle do not intersect, None is returned.
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

		if almosteq(r5, 0.0):
			# Line is parallel to facet
			if almosteq(dot(y, self.normal), 0.0):
				raise NotImplementedError('Segment seg is in plane of facet')
			return None

		q2 = tuple(v / r5 for v in q2)

		# Invert the QR factorization
		y = tuple(dot(y, b) for b in (q0, q1, q2))
		v = y[2] / r5
		u = (y[1] - r4 * v) / r2
		t = (y[0] - r3 * v - r1 * u) / r0

		if all(0 <= l <= 1 for l in (v, u, t)) and 0 <= u + t <= 1:
			# v is the fraction of segment length
			# t and u are normal barycentric coordinates in triangle
			return v * seg.length, t, u, 1 - t - u
		else:
			# Intersection is not in segment or triangle
			return None


class Box3D(object):
	'''
	A representation of an axis-aligned 3-D bounding box.
	'''
	def __init__(self, lo, hi):
		'''
		Initialize a 3-D box with extreme corners lo and hi.
		'''
		_checkdims(lo)
		_checkdims(hi)

		# Store an immutable, float copy of the corners
		self._lo = tuple(float(x) for x in lo)
		self._hi = tuple(float(x) for x in hi)

		eps = sys.float_info.epsilon
		if any(hv < lv for lv, hv in izip(self._lo, self._hi)):
			raise ValueError('Coordinates of hi must be no less than those of lo')

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
		# Check for sane values
		_checkdims(c)
		if any(x < 1 for x in c):
			raise ValueError('Grid dimensions must each be at least 1')

		# Set the cell count
		self._ncell = tuple(int(x) for x in c)
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
			self._cell = tuple(d / float(l)
					for d, l in izip(self.length, self.ncell))
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
		p = tuple((c - l) / d for c, l, d in izip((x,y,z), self.lo, self.cell))

		if not frac:
			return tuple(int(c) for c in p)

		return p

	def cell2cart(self, i, j, k):
		'''
		Convert the (possibly fractional) 3-D cell-index coordinates
		(i, j, k), defined by the box bounds and ncell property, into
		Cartesian coordinates.
		'''
		return tuple(n * d + l for n, l, d in izip((i,j,k), self.lo, self.cell))

	def getCell(self, c):
		'''
		Return a Box3D representing the cell that contains 3-D index c
		based on the grid defined by the ncell property.

		If c does not contain integer types, the types will be
		truncated. Cells outside the bounds of this box are allowed.
		'''
		_checkdims(c)
		c = tuple(int(cv) for cv in c)
		lo = self.cell2cart(*c)
		hi = self.cell2cart(*(cv + 1 for cv in c))
		return Box3D(lo, hi)

	def allIndices(self):
		'''
		Return a generator that produces every 3-D cell index within
		the grid defined by the ncell property in row-major order.
		'''
		return iproduct(*(xrange(n) for n in self.ncell))

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
		(alx, aly, alz), (ahx, ahy, ahz) = self.lo, self.hi
		(blx, bly, blz), (bhx, bhy, bhz) = b.lo, b.hi

		return not (ahx < blx or alx > bhx or ahy < bly or
				aly > bhy or ahz < blz or alz > bhz)

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
		Octree.addleaves.

		When created, descendants will be accumulated in the children
		property of this object. For levels greater than 0, children is
		a dictionary that maps octant indices (i, j, k), where each
		index can take the value 0 or 1, to an Octree instance that
		represents a branch at (level - 1). For level-0 trees, children
		is just a set of arbitrary objects.
		'''
		self.level = int(level)
		if self.level != level:
			raise TypeError('Argument level must be an integer')
		if self.level < 0:
			raise ValueError('Argument level must be nonnegative')

		self.rootbox = Box3D(box.lo, box.hi)

		# At level 0, children (leaves) are stored in a set
		if self.level < 1:
			self.children = set()
			return

		# Subdivide the root into octants
		self.rootbox.ncell = (2, 2, 2)
		# Children of nonzero levels are stored in a dictionary
		self.children = { }

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
		From the iterable leaves, populate the children of all level-0
		branches in the Octree hierarchy according to the value of the
		given predicate. If any branches are missing from the tree,
		they will be created as necessary. Leaf objects must be
		hashable.

		The predicate must be a callable that takes two positional
		arguments, box and leaf, and returns True iff the specified
		Box3D box contains the leaf. The predicate will be called for
		boxes at every level of the tree while drilling down.

		If multibox is True, all level-0 boxes that satisfy the
		predicate for a given leaf will record the leaf as a child.
		Otherwise, only the first box to satisfy the predicate will own
		the box.  The tree is walked depth-first, with children
		encountered in the order determined by Box3D.allIndices.

		If no box contains an entry in leaves, that entry will be
		silently ignored.

		This method returns True if any leaf was added to the tree,
		False otherwise.
		'''
		rbox = self.rootbox
		added = False

		for leaf in leaves:
			# Check whether the leaf belongs in this tree
			if not predicate(rbox, leaf): continue

			if self.level < 1:
				# Just add the children at the finest level
				self.children.add(leaf)
				added = True
				continue

			# At higher levels, try to add the leaf to children
			for idx in rbox.allIndices():
				# Grab or create a child tree
				try:
					ctree = self.children[idx]
				except KeyError:
					cbox = rbox.getCell(idx)
					ctree = Octree(self.level - 1, cbox)

				# Add the leaf to the child tree, if possible
				if ctree.addleaves((leaf,), predicate, multibox):
					added = True
					# Make sure the child is recorded
					self.children.setdefault(idx, ctree)
					if not multibox: break

		return added

	def search(self, boxpred, leafpred=None):
		'''
		Perform a depth-first search of the tree to identify matching
		leaves. A leaf is said to match the search iff the value of
		Boolean value of leafpred(leaf) is True and the Boolean value
		of boxpred(box) is True for the level-0 box that contains the
		leaf and for all of its ancestors.

		The order in which children are followed is arbitrary.

		The callable boxpred should take a single Box3D argument, which
		will be the root of some branch of the Octree. If the Boolean
		value of boxpred(box) for some box, the branch rooted on the
		box will be further searched. Otherwise, searching will
		terminate without checking descendants.

		The optional callable leafpred should take as its sole argument
		a leaf object previously assigned to the tree using the method
		Octree.addleaves. If leafpred is not defined, the default
		implementation returns True for every leaf.

		The return value will be a dictionary mapping all leaf objects
		that match the search to the value returned by leafpred(leaf).
		'''
		# Match is empty if the box predicate fails
		if not boxpred(self.rootbox): return { }

		if self.level > 0:
			# Recursively check branches
			results = { }
			for ctree in self.children.itervalues():
				results.update(ctree.search(boxpred, leafpred))
			return results

		# With no leaf predicate, all leaves match
		if not leafpred: return { c: True for c in self.children }

		# Otherwise, filter leaves by the leaf predicate
		results = { }
		for c in self.children:
			lp = leafpred(c)
			if lp: results[c] = lp
		return results
