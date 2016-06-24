'''
Utilities for image manipulation.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, math


class MultiplaneProjector(object):
	'''
	A class that manages a group of image slices and a PinholeCamera
	instance, projecting and compositing the slices onto the camera image
	plane. Slices are referenced externally by user-provided keys.

	Bilinear interpolation of source images during projections is
	accomplished using scipy.interpolate.RegularGridInterpolator instances.
	'''
	def __init__(self, *args):
		'''
		Create a multiplane projector using an appropriately specified
		camera.

		If args are empty or a single argument is None, the projector
		will have no associated camera.

		If one argument is specified and it is an instance of
		PinholeCamera, a reference will be stored in the camera
		property of this object.

		Otherwise, self.camera = PinholeCamera(*args).
		'''
		# Initialize an empty list of image slices and cached projections
		self._slices = { }
		self._imgcache = { }

		la = len(args)

		if not la or (la == 1 and args[0] is None):
			self._camera = None
		elif la == 1 and isinstance(args[0], PinholeCamera):
			self._camera = args[0]
		else:
			self._camera = PinholeCamera(*args)


	@property
	def camera(self):
		'''
		The camera instance used to capture the composite image.
		'''
		return self._camera


	@camera.setter
	def camera(self, cam):
		'''
		Assign the PinholeCamera instance cam to the camera property.
		'''
		if cam is not None and not isinstance(cam, PinholeCamera):
			raise ValueError('Property camera must be a PinholeCamera instance or None')

		self._camera = cam
		# Clear the image cache because new projections will be required
		self._imgcache = { }


	def boundingBox(self):
		'''
		A float array of shape (8, 3) that contains, along the rows,
		the world coordinates of the cube that bounds all registered
		image slices.
		'''
		x, y, z = [], [], []
		nax = np.newaxis

		if not len(self._slices):
			return np.array([[]], dtype=float)

		for _, (m, n), (dm, dn), o in self._slices.itervalues():
			# Build the four corners of each slice
			i, j = np.array([[0]*2 + [m-1]*2, [0, n-1]*2], dtype=float)
			crd = o[nax,:] + i[:,nax] * dm[nax,:] + j[:,nax] * dn[nax,:]
			# Add the coordinates
			x.extend(crd[:,0])
			y.extend(crd[:,1])
			z.extend(crd[:,2])

		lx, hx = min(x), max(x)
		ly, hy = min(y), max(y)
		lz, hz = min(z), max(z)

		return np.array([[lx, ly, lz], [lx, hy, lz], [hx, ly, lz],
			[hx, hy, lz], [lx, ly, hz], [lx, hy, hz],
			[hx, ly, hz], [hx, hy, hz]], dtype=float)


	def delslice(self, key, ignore_error=True):
		'''
		Attempt to remove a previously registered slice with the
		provided key from the projector. If ignore_error is True, no
		error will be raised if no image exists for the given key.
		Otherwise, a KeyError will be raised if no image exists.
		'''
		# Always ignore missing keys in the cache
		self._imgcache.pop(key, None)
		# Ignore missing keys in the slice list if desired
		try: self._slices.pop(key)
		except KeyError as k:
			if not ignore_error: raise k


	def setslice(self, key, img, basis, origin):
		'''
		Register the provided image img, with a basis and origin as
		processed by self.camera.validatePixelGrid (or, if self.camera
		is None, PinholeCamera.validatePixelGrid) for projection and
		associate the image with the provided key.

		The image should be a 2-D Numpy array of floats.
		'''
		from scipy.interpolate import RegularGridInterpolator

		img = np.asarray(img, dtype=float)

		try: m, n = img.shape
		except ValueError:
			raise ValueError('Argument img should be a 2-D array')

		# Validate the image grid
		if self.camera is None:
			validator = PinholeCamera.validatePixelGrid
		else:
			validator = self.camera.validatePixelGrid
		m, n, basis, origin = validator(m, n, basis, origin)

		# Build the image interpolator
		idim = np.arange(float(m)), np.arange(float(n))
		intp = RegularGridInterpolator(idim, img, bounds_error=False)

		# Store the interpolator and grid information
		self._slices[key] = (intp, (m, n), basis, origin)

		# Remove any existing cached image for this key
		self._imgcache.pop(key, None)


	@staticmethod
	def _validateImageSize(m, n):
		'''
		Ensure that m and n are both positive integers, returning m and
		n as integers.
		'''
		mi, ni = int(m), int(n)
		if mi != m or ni != n or mi < 1 or ni < 1:
			raise ValueError('Image sizes (m, n) must be positive integers')
		return mi, ni


	def getslice(self, key, psize, f):
		'''
		Get, as (img, depth), the projected slice image and associated
		depth map for the slice identified by key. The image will have
		size psize and a focal length f.
		
		The local image cache is first checked for an existing image
		and depth map with matching size and focal length. If no valid
		image or depth map exists, both will be recomputed and stored
		in the cache (possibly replacing a cached images for different
		psize and f values).
		'''
		try:
			intp, ssize, basis, origin = self._slices[key]
		except (KeyError, ValueError):
			raise KeyError('No valid image exists for key "%s"' % key)

		cam = self.camera
		if cam is None:
			raise ValueError('Cannot project image when self.camera is None')

		tm, tn = self._validateImageSize(*psize)

		try:
			img, depth, (cm, cn), cf = self._imgcache[key]
		except (KeyError, ValueError):
			pass
		else:
			if (tm, tn) == (cm, cn) and abs(cf - f) < np.finfo(float).eps:
				return img, depth

		# Need to recompute the slice
		crds, depth = cam.rproject(psize, ssize, basis, origin, f)
		img = intp(crds)

		# Cache these results
		self._imgcache[key] = img, depth, (tm, tn), f
		return img, depth


	def iterkeys(self):
		'''
		Return an iterator of slice keys registered with the projector.
		'''
		return self._slices.iterkeys()


	def slicecount(self):
		'''
		Return the number of slices registered with the projector.
		'''
		return len(self._slices)


	def composite(self, psize, f = None):
		'''
		Return a composite image and depth map, each with a size psize
		and a focal length f, of all registered planes for the current
		camera.

		If f is None, it will be be optimized so that all projections
		fill the image plane with no clipping.
		'''
		cam = self.camera
		if cam is None:
			raise ValueError('Cannot composit image when self.camera is None')

		tm, tn = self._validateImageSize(*psize)

		if f is None:
			ncrd = cam.project(self.boundingBox())
			try: f = cam.optimumFocalLength(ncrd, tm, tn)
			except ValueError: f = 1.

		output = np.empty((tm, tn), dtype=float)
		cdepth = np.empty((tm, tn), dtype=float)

		output[:,:] = float('nan')

		# Shorten the function names
		_and = np.logical_and
		_not = np.logical_not
		_or = np.logical_or
		_isnan = np.isnan

		for key in self.iterkeys():
			img, depth = self.getslice(key, (tm, tn), f)

			onan = _isnan(output)
			inan = _isnan(img)

			# All NaNs can be replaced with valid image pixels
			idx = _and(onan, _not(inan))
			output[idx] = img[idx]
			cdepth[idx] = depth[idx]

			# Look for new pixels closer than existing pixels
			# Restrict subsequent changes to previously valid pixels
			idx = _and(_not(onan), _and(_not(inan), depth < cdepth))
			output[idx] = img[idx]
			cdepth[idx] = depth[idx]

		return output, _isnan(output).choose(cdepth, float('nan'))


class PinholeCamera(object):
	'''
	A class that projects planar images onto an image plane with an
	arbitrary 3-D orientation using a pinhole camera model.
	'''
	@staticmethod
	def imgbounds(m, n):
		'''
		Given dimensions (m, n) for an image plane, return bounds

			(lm, ln, hm, hn),

		where (lm, ln) is the center of the lower-left pixel in image
		coordinates, and (hm, hn) is the center of the upper-right
		pixel in image coordinates.

		The coordinates are symmetric about the origin.
		'''
		sm, sn = [-v / 2. + 0.5 for v in (m, n)]
		return (sm, sn, sm + m - 1, sn + n - 1)


	@staticmethod
	def axes(orientation):
		'''
		Create and return the camera axes (xp, yp, zp) for a pinhole
		camera with the given 3-vector orientation in the standard
		orthonormal basis (X, Y, Z). The orientation vector points from
		the pinhole to the center of the imaging target.

		The axis zp is always parallel to the orientation. The axis xp
		is perpendicular to both Z and zp (i.e., xp lies in the X-Y
		plane), while yp is perpendicular to xp and zp.

		In the special case where the orientation and zp are equal to
		Z, xp will be X and yp will be Y. When orientation and zp are
		equal to -Z, xp will be Y and yp will be X.

		Whenever yp is not in the X-Y plane, the signs of xp and yp
		will be chosen so that the Z component of yp is positive.

		Arithmetic is done in native Python floats.
		'''
		eps = np.finfo(float).eps

		zx, zy, zz = [float(ov) for ov in orientation]
		zl = math.sqrt(zx**2 + zy**2 + zz**2)

		if zl <= eps:
			raise ValueError('Orientation must not be 0')

		zp = [zx / zl, zy / zl, zz / zl]
		zx, zy, zz = zp

		zxy = math.sqrt(zx**2 + zy**2)

		if zxy <= eps: xp = [1., 0., 0.]
		else: xp = [-zy / zxy, zx / zxy, 0.]

		xx, xy, _ = xp

		yp = [-zz * xy, zz * xx, zx * xy - zy * xx]

		if yp[2] < 0:
			# Flip x and y to ensure upward-aimed yp
			yp = [-yp[0], -yp[1], -yp[2]]
			xp = [-xx, -xy, 0.]

		return xp, yp, zp


	def __init__(self, loc, orientation=None):
		'''
		Initialize a pinhole camera with the pinhole at the 3-vector
		location loc = (x, y, z) and an orientation vector

			orientation = (nx, ny, nz)

		that points from loc toward the center of the target. The
		length of the orientation vector is ignored. If orientation is
		None, it is assumed that the camera points at the origin, so

			orientation = (-x, -y, -z).
		'''
		self.center = np.asarray(loc, dtype=float).squeeze()
		if self.center.ndim != 1 or len(self.center) != 3:
			raise ValueError('Camera location must be a 3-vector')

		if orientation is None:
			orientation = self.center

		# The rotation matrix has orthonormal camera basis along rows
		self.rotmat = np.array(self.axes(orientation), dtype=float)


	@classmethod
	def validatePixelGrid(cls, m, n, basis, origin):
		'''
		Ensure that the parameters m, n, basis and origin describe a
		valid pixel grid for (m, n) pixels. Validation checks are:

		1. m and n should be positive integers,

		2. basis should be a 2-by-3 array (see Note 1) wherein the two
		   rows are orthogonal, the first row gives the coordinate
		   offset between pixel (i+1,j) and (i,j) and the second row
		   gives the coordinate offset between pixel (i,j+1) and (i,j),

		3. origin is a 3-vector (see Note 2) specifying the world
		   coordinates of pixel (0, 0).

		These criteria imply that the coordinates pixel (i, j) are

			crd(i, j) = origin + i * basis[0] + j * basis[1].

		Return values are m and n as integers, basis as a Numpy float
		array of shape (2,3), and origin as a Numpy float array of
		shape (3,).

		*** Note 1 ***

		The basis argument can also be a 2-character string from the
		take from the (case-insensitive) set

			{ 'x', 'y', 'z' }**2 - { 'xx', 'yy', 'zz' }

		where the first character determines the first basis row and
		the second character determines the second basis row according
		to the map

			{ 'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1] }.

		*** Note 2 ***

		The origin argument can take one of two special values for
		convenience:

		1. None or empty sequence: Use a "natural" origin where, for

			lm, ln, hm, hn = cls.imgbounds(m, n),

		   origin = lm * basis[0] + ln * basis[1], or

		2. Scalar f or 1-vector [f]: Use an offset "natural" origin
		   where, for

		   	lm, ln, hm, hn = cls.imgbounds(m, n),

		   origin = lm * basis[0] + ln * basis[1] + f * nrm for

			nrm = np.cross(basis[0], basis[1]) / np.norm(nrm).

		   Note that the normal nrm is normalized even if basis[0] or
		   basis[1] are not. Also note that the direction of nrm
		   follows the right-hand rule so, e.g., the normal for basis
		   'xz' points in the -y direction.
		'''
		mi, ni = int(m), int(n)
		if mi != m or ni != n or mi < 1 or ni < 1:
			raise ValueError('Arguments m, n must have positive integer values')

		if isinstance(basis, basestring):
			basis = basis.lower()
			try:
				dm, dn = basis
			except ValueError:
				raise ValueError('String argument basis must contain exactly 2 characters')
			# Map the characters to appropriate basis functions
			bmap = dict(zip('xyz', np.eye(3, dtype=float)))
			try:
				basis = np.array([bmap[dm], bmap[dn]], dtype=float)
			except KeyError:
				raise ValueError('Characters in string argument basis must be in set { "x", "y", "z" }')
		else:
			basis = np.asarray(basis, dtype=np.float)
			if basis.shape != (2, 3):
				raise ValueError('Array argument basis must have shape (2, 3)')

		eps = np.finfo(float).eps
		if abs(np.dot(basis[0], basis[1])) > eps:
			raise ValueError('Rows of basis matrix must be orthogonal')
		if any(d < eps for d in np.sum(basis, axis=-1)):
			raise ValueError('Rows of basis matrix must not be 0')

		if origin is None:
			origin = np.array([], dtype=float)
		else: origin = np.asarray(origin, dtype=float)

		if origin.ndim == 0:
			origin = origin[np.newaxis]
		elif origin.ndim != 1:
			raise ValueError('Argument origin must be None, scalar, 1-vector, or 3-vector')

		if len(origin) <= 1:
			try:
				f = origin[0]
				nrm = np.cross(basis[0], basis[1])
				nrm /= np.sqrt(np.sum(nrm**2))
			except IndexError:
				f, nrm = 0, 0
			lm, ln, hm, hn = cls.imgbounds(m, n)
			origin = lm * basis[0] + ln * basis[1] + f * nrm
		elif len(origin) != 3:
			raise ValueError('Argument origin must be None, scalar, 1-vector, or 3-vector')

		return mi, ni, basis, origin


	def project(self, pts):
		'''
		Project the N-dimensional array pts of world coordinates into
		normalized, homoegeneous camera coordinates. The output will
		have the same shape as the input.

		The coordinates are specified along the final axis, so
		pts.shape[-1] == 3 and output.shape[-1] == 3.
		'''
		pts = np.asarray(pts, dtype=float)

		if pts.ndim < 1 or pts.shape[-1] != 3:
			raise ValueError('Input pts must be at least 1-D with a final axis of length 3')

		# Pad the origin to perform camera transform
		axpad = [np.newaxis] * (pts.ndim - 1) + [slice(None)]

		# Shift origin and rotate to camera coordinates
		return np.dot(pts - self.center[axpad], self.rotmat.T)


	def optimumFocalLength(self, crd, m, n):
		'''
		Given an N-by-3 array of homogeneous, normalized image
		coordinates crd and a desired image size of (m, n) pixels,
		where the image is centered behind the camera pinhole, return
		the longest integer focal length of the camera that completely
		contains all image coordinates.
		'''
		crd = np.asarray(crd, dtype=float)
		if crd.ndim == 1: crd = crd[np.newaxis,:]

		if crd.ndim != 2 or crd.shape[1] != 3:
			raise ValueError('Argument crd must be a 3-vector or an M-by-3 array')

		# Find image bounds
		lx, ly, hx, hy = self.imgbounds(m, n)

		# Find target bounds in image coordinates
		ncrd = crd[:,:2] / crd[:,2,np.newaxis]
		mnx, mny = np.min(ncrd, axis=0)
		mxx, mxy = np.max(ncrd, axis=0)

		if mnx < 0 and lx >= 0 or mny < 0 and ly >= 0:
			raise ValueError('Unable to project negative target coordinates into positive image plane')
		elif mxx > 0 and hx <= 0 or mxy > 0 and hy <= 0:
			raise ValueError('Unable to project positive target coordinates into negative image plane')

		f = [ ]

		# Figure focal bounds for negative coordinates
		if mnx < 0:
			f.append(lx / mnx)
		if mny < 0:
			f.append(ly / mny)

		# Figure focal bounds for positive coordinates
		if mxx > 0:
			f.append(hx / mxx)
		if mxy > 0:
			f.append(hy / mxy)

		try:
			f = float(int(min(f)))
			if f <= 0: raise ValueError
		except ValueError:
			raise ValueError('Could not find optimum positive focal length')
		else:
			return f


	def rproject(self, imgshape, srcshape, basis, origin, f=None):
		'''
		Given an image plane containing imgshape = (tm, tn) pixels
		centered on the optical axis, return (coords, depth), where the
		array coords has shape (tm, tn, 2) and slice [i', j', :] yields
		the fractional pixel coordinates of a point in a source plane
		that projects onto image pixel (i', j'). The source plane is
		defined according to the rules of

			self.validatePixelGrid(m, n, basis, origin)

		for (m, n) = srcshape.

		The array depth has shape (tm, tn) and maps each image pixel to
		its projective distance to the source plane.

		Any source coordinates that are outside of the image bounds
		will be clipped to the range [-1, m] (for the i-coordinate) or
		[-1, n] (for the j-coordinate). A degenerate geometry (i.e., a
		source plane parallel to some of the projection rays) will
		result in both coords and depth maps containing all -1 values.

		The camera is assumed to have focal length f. If f is None, it
		is determined with self.optimumFocalLength.
		'''
		tm, tn = imgshape
		m, n = srcshape

		if any(v < 1 for v in (tm, tn, m, n)):
			raise ValueError('Pixel dimension tm, tn, m, and n must be positive')

		nax = np.newaxis

		m, n, (dm, dn), origin = self.validatePixelGrid(m, n, basis, origin)
		normal = np.cross(dm, dn)

		if f is None:
			i, j = np.array([[0]*2 + [m-1]*2, [0, n-1]*2], dtype=float)
			crd = origin[nax,:] + i[:,nax] * dm[nax,:] + j[:,nax] * dn[nax,:]
			ncrd = self.project(crd)

			try: f = self.optimumFocalLength(ncrd, tm, tn)
			except ValueError: f = 1.

		# Rotate the source origin and normal into camera coordinates
		p0 = self.project(origin)
		normal = np.dot(self.rotmat, normal)

		eps = np.finfo(float).eps

		# Find numerator for intersection parameter
		# In camera coordinates, the line "origin" is 0
		pn = np.dot(p0, normal)

		if abs(pn) < eps:
			# In degenerate cases, return all -1s
			return (-np.ones((tm, tn, 2), dtype=float),
					-np.ones((tm, tn), dtype=float))

		# Build a dense grid for projection
		lx, ly, hx, hy = self.imgbounds(tm, tn)
		l = np.empty((tm, tn, 3), dtype=float)
		l[:,:,0] = np.arange(lx, hx + 1, dtype=float)[:,nax]
		l[:,:,1] = np.arange(ly, hy + 1, dtype=float)[nax,:]
		# Image coordinates have z height of the focal length
		l[:,:,2] = -f

		# Project the image coordinates into the source plane
		l *= (pn / np.dot(l, normal))[:,:,nax]
		# Compute the distance from source plane to aperture
		depth = np.sqrt(np.sum(l**2, axis=-1))

		# Transform intersection points into source coordinates
		crds = np.dot(l, self.rotmat) + self.center[nax,nax,:]
		# Now convert source coordinates to pixel coodinates
		crds -= origin[nax,nax,:]
		# This works because dm and dn are be orthogonal
		# Clip out-of-bounds values
		ic = (np.dot(crds, dm) / np.sum(dm**2)).clip(-1, m)
		jc = (np.dot(crds, dn) / np.sum(dn**2)).clip(-1, n)

		# Combine the pixel coordinates along the final axis
		crds = np.array([ic, jc], dtype=float).transpose(1, 2, 0)

		return crds, depth
