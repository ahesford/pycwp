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
from .cutil import cross, dot, norm, almosteq
from .cytools.boxer import Segment3D, Triangle3D, Box3D

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

	def branchForKey(self, *key):
		'''
		Query this tree for the branch (an Octree object) identified by
		the provided key. The key, which must be a sequence, can be
		provided as unpacked arguments or as a single argument.

		The length of the key must be a multiple of three. If the key
		is empty, self is returned. If the key is not empty, it is
		split as

			child = key[:3]
			subkey = key[3:]

		and a recursive result is returned by calling

			self.children[child].branchForKey(subkey).

		If a branch for the given child key does not exist, but the
		child key represents a valid octant, the branch is created.

		If the child key does not represent a valid octant, either
		because the key has a length other than 0 or 3, or because the
		child key contains values other than 0 or 1, a KeyError will be
		raised. At level 0, any nonempty key will raise a KeyError.
		'''
		# Treat a single argument as a packed index
		if len(key) == 1: key = key[0]

		try:
			child = tuple(key[:3])
			subkey = key[3:]
		except TypeError:
			raise KeyError('Single argument must be a sequence')

		if not len(child): return self

		if self.level < 1:
			raise KeyError('Key length greater than tree depth')
		elif len(child) != 3:
			raise KeyError('Key length must be a multiple of three')

		if set(child).difference({ 0, 1 }):
			raise KeyError('Indices of key must be 0 or 1')

		try: ctree = self.children[child]
		except KeyError:
			cbox = self.rootbox.getCell(*child)
			ctree = Octree(self.level - 1, cbox)
			self.children[child] = ctree

		return ctree.branchForKey(subkey)

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
		the box. The tree is walked depth-first, with children
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
				ctree = self.branchForKey(idx)
				# Add the leaf to the child tree, if possible
				if ctree.addleaves((leaf,), predicate, multibox):
					added = True
					if not multibox: break

		return added

	def mergeleaves(self, leaves):
		'''
		For each key-value pair leaves, a mapping from branch keys to
		sets of leaf objects (in the same form produced by the method
		Octree.getleaves), add all leaf objects in the set to the
		level-0 branch indicated by the corresponding key.

		A KeyError will be raised wihtout adding leaves if any keys
		in the mapping fail to specify valid level-0 Octree branches.
		Some intermediate branches may be added even if leaves are not
		added, but the added branches will be empty in this case.

		The same leaf object will be added to multiple level-0 branches
		if the object is specified for multiple keys in the leaves
		mapping.
		'''
		# Produce a list of (child, leaf-set) pairs for validation
		bpairs = [ (self.branchForKey(key), set(lset))
				for key, lset in leaves.iteritems() ]

		# Ensure all branches are at level 0
		if any(branch.level for branch, lset in bpairs):
			raise KeyError('Branch keys in leaves mapping must idenify level-0 children')

		# Add all of the children
		for branch, lset in bpairs: branch.children.update(lset)

	def getleaves(self):
		'''
		Return a mapping from addresses to leaf sets such that, for a
		key-value pair (key, leaves) in the mapping, the branch
		returned by self.branchForkey(key) is a level-0 Octree and
		self.branchForKey(key).children == leaves.
		'''
		if self.level < 1:
			# At the lowest level, The address is empty
			return { (): set(self.children) }

		# Build the mapping up for all children
		return { tuple(key) + ck: cv
			for key, ctree in self.children.iteritems()
			for ck, cv in ctree.getleaves().iteritems() }

	def search(self, boxpred, leafpred=None, leafcache=None):
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

		If leafcache is provided, it must be a dictionary. When
		attempting to match leaves in the search, the value of
		leafcache[leaf] will be used as a substitute for the value of
		leafpred(leaf) whenever possible. If leaf is not in leafcache,
		the value leafpred(leaf) will be assigned to leafcache[leaf].
		This capability is useful to avoid redundant match tests for
		leaves added in "multibox" mode and guarantees that
		leafpred(leaf) will be evaluated at most once for each leaf.

		The return value will be a dictionary mapping all leaf objects
		that match the search to the value returned by leafpred(leaf)
		or leafcache[leaf].
		'''
		# Match is empty if the box predicate fails
		if not boxpred(self.rootbox): return { }

		if self.level > 0:
			# Recursively check branches
			results = { }
			for ctree in self.children.itervalues():
				results.update(ctree.search(boxpred, leafpred, leafcache))
			return results

		# With no leaf predicate, all leaves match
		if not leafpred: return { c: True for c in self.children }

		# Otherwise, filter leaves by the leaf predicate
		results = { }
		for c in self.children:
			if leafcache is not None:
				try:
					lp = leafcache[c]
				except KeyError:
					lp = leafpred(c)
					leafcache[c] = lp
			else: lp = leafpred(c)
			if lp: results[c] = lp
		return results
