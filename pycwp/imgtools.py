'''
Utilities for image manipulation.
'''

# Copyright (c) 2015 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.

import numpy as np, math


class PinholeCamera(object):
	'''
	A class that projects planar images onto an image plane with an
	arbitrary 3-D orientation using a pinhole camera model.
	'''
	@classmethod
	def imgcoords(cls, m, n):
		'''
		Given dimensions (m, n) for a desired image plane, return as
		(x, y) an ogrid representation of image coordinates that are
		symmetric about the origin. The raster order is such that, for

			lx, ly, hx, hy = cls.imgbounds(m, n),

		pixel (0, 0) has coordinates (lx, ly) and pixel (m - 1, n - 1)
		has coordinates (hx, hy).
		'''
		lx, ly, hx, hy = cls.imgbounds(m, n)
		return np.ogrid[lx:hx+1,ly:hy+1]


	@staticmethod
	def imgbounds(m, n):
		'''
		Given dimensions (m, n) for an image plane, return bounds

			(lx, ly, hx, hy),

		where (lx, ly) is the center of the lower-left pixel in image
		coordinates, and (hx, hy) is the center of the upper-right
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


	@staticmethod
	def _validatePixelGrid(origin, dm, dn):
		'''
		Ensure that origin, dm, and dn are 3-vectors as Numpy arrays
		and that dm and dn are orthogonal and have non-zero length.

		Returns origin, dm, and dn as Numpy float arrays.
		'''
		origin = np.asarray(origin, dtype=float).squeeze()
		dm = np.asarray(dm, dtype=float).squeeze()
		dn = np.asarray(dn, dtype=float).squeeze()

		if any(v.ndim != 1 or len(v) != 3 for v in (origin, dm, dn)):
			raise ValueError('Points origin, dm, and dn must be 3-D vectors')

		eps = np.finfo(float).eps

		if abs(np.dot(dm, dn)) > eps:
			raise ValueError('Pixel offsets dm and dn must be orthogonal')

		if any(sum(v**2 for v in d) <= eps for d in (dm, dn)):
			raise ValueError('Pixel offsets must not be 0')

		return origin, dm, dn


	def normalizedCoords(self, *args):
		'''
		Valid signatures:

		self.normalizedCoords(pts) or
		self.normalizedCoords(m, n, origin, dm, dn)

		In the single-argument form, pts should be an L-by-3 array of
		spatial coordinates. The return value is an L-by-3 array of
		homogeneous, normalized image coordinates for each of the L
		input coordinates.

		In the multiple-argument form, for an original image of shape
		(m, n), where the pixel at index (i [< m], j [< n]) has spatial
		coordinates

			crd(i,j) = origin + i * dm + j * dn

		for 3-D points origin, dm and dn, return a Numpy array icrd of
		shape (m, n, 3), where icrd[i,j,:] holds the homogeneous,
		normalized image coordinates for pixel (i, j).
		'''
		if len(args) == 1:
			# Make sure the coordinates are a copy
			crd = np.array(args[0], dtype=float)

			if crd.ndim != 2 or crd.shape[1] != 3:
				raise ValueError('Single argument pts must be an L-by-3 array')

			# Subtract the camera center...
			nax = np.newaxis
			crd -= self.center[nax,:]

			return np.dot(crd, self.rotmat.T)
		elif len(args) != 5:
			raise TypeError('Invalid argument list')

		m, n, origin, dm, dn = args

		if m < 1 or n < 1:
			raise ValueError('Image dimensions (m, n) must be positive')

		# Use arrays for fancy slicing
		origin, dm, dn = self._validatePixelGrid(origin, dm, dn)

		# Build the (m,n,3) spatial coordinate grid
		i, j = np.ogrid[:float(m),:float(n)]
		nax = np.newaxis
		crd = (origin[nax, nax, :]
				+ i[:, :, nax] * dm[nax, nax, :]
				+ j[:, :, nax] * dn[nax, nax, :])

		# Translate the coordinate origin to the camera center
		crd -= self.center[nax, nax, :]

		# Rotate into homogeneous, normalized coordinates
		return np.dot(crd, self.rotmat.T)


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


	def revcoords(self, tm, tn, m, n, origin, dm, dn, f=None):
		'''
		Given an image plane containing (tm, tn) pixels centered on the
		optical axis, return (coords, depth), where the array coords
		has shape (tm, tn, 2) and slice [i', j', :] yields the
		fractional pixel coordinates of a point in a source plane that
		projects onto image pixel (i', j'). The source plane is defined
		as a grid of (m, n) pixels such that source pixel (i, j) has
		coordinates

			crd(i, j) = origin + i * dm + j * dn

		for 3-vectors origin, dm, and dn.

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
		if any(v < 1 for v in (tm, tn, m, n)):
			raise ValueError('Pixel dimension tm, tn, m, and n must be positive')

		nax = np.newaxis

		origin, dm, dn = self._validatePixelGrid(origin, dm, dn)
		normal = np.cross(dm, dn)

		if f is None:
			i, j = np.array([[0]*2 + [m-1]*2, [0, n-1]*2], dtype=float)
			crd = origin[nax,:] + i[:,nax] * dm[nax,:] + j[:,nax] * dn[nax,:]
			ncrd = self.normalizedCoords(crd)

			try: f = self.optimumFocalLength(ncrd, tm, tn)
			except ValueError: f = 1.

		# Rotate the source origin and normal into camera coordinates
		p0 = self.normalizedCoords([origin])[0]
		normal = np.dot(self.rotmat, normal)

		eps = np.finfo(float).eps

		# Find numerator for intersection parameter
		# In camera coordinates, the line "origin" is 0
		pn = np.dot(p0, normal)

		# In degenerate cases, return all -1s
		if abs(pn) < eps:
			negone = -np.ones((tm, tn, 2), dtype=float)
			return negone, negone

		# Build a dense grid for projection
		x, y = self.imgcoords(tm, tn)
		l = np.empty((tm, tn, 3), dtype=float)
		l[:,:,0] = x
		l[:,:,1] = y
		# Image coordinates have z height of the focal length
		l[:,:,2] = -f

		# Project the image coordinates into the source plane
		l *= (pn / np.dot(l, normal))[:,:,nax]
		# Compute the distance from source plane to aperture
		depth = np.sqrt(l[:,:,0]**2 + l[:,:,1]**2 + l[:,:,2]**2)

		# Transform intersection points into source coordinates
		crds = np.dot(l, self.rotmat) + self.center[nax,nax,:]
		# Now convert source coordinates to pixel coodinates
		crds -= origin[nax,nax,:]
		# This works because dm and dn are be orthogonal
		# Clip out-of-bounds values
		ic = (np.dot(crds, dm) / sum(v**2 for v in dm)).clip(-1, m)
		jc = (np.dot(crds, dn) / sum(v**2 for v in dn)).clip(-1, n)

		# Combine the pixel coordinates along the final axis
		crds = np.array([ic, jc], dtype=float).transpose(1, 2, 0)

		return crds, depth


	def project(self, tm, tn, img, origin, dm, dn, f=None):
		'''
		Project, onto a camera image of shape (tm, tn) centered on the
		camera axis, the source image img. Returns (pixmap, depthmap),
		where pixmap is the projected image and depthmap maps each
		projected pixel to a distance from the corresponding target
		value to the camera pinhole. Linear interpolation is done using
		the function scipy.interpolate.griddata.

		Arguments origin, dm, and dn should be 3-vectors such that
		pixel (i, j) in the source img has spatial coordinates

			crd(i, j) = origin + i * dm + j * dn

		(e.g., origin, dm, and dn have the same interpretation as for
		self.normalizedCoords).

		If f is not None, it should be a positive number which will be
		the focal length of the camera for the projection. If f is
		None, it will be determined with self.optimumFocalLength.

		If self.optimumFocalLength fails for any reason, the focal
		length will be assumed to be unity.

		A front-image camera is assumed, so the coordinates will be in
		the proper orientation.
		'''
		from scipy.interpolate import griddata

		# Try to identify a single image or list of images
		try:
			img = np.asarray(img, dtype=float)
			if img.ndim != 2: raise ValueError
		except ValueError:
			raise ValueError('Argument img must be a 2-D array')

		# Determine the source shape and coordinates
		m, n = img.shape
		crd = self.normalizedCoords(m, n, origin, dm, dn)
		crd = crd.reshape((m * n, 3), order='C')
		# Negate the focus to make a front-image camera
		crd[:,-1] = -crd[:,-1]

		# Determine the focal length if necessary
		if f is None:
			try: f = self.optimumFocalLength(crd, tm, tn)
			except ValueError: f = 1.0

		# Homogeneous to image coordinates, with focal length
		ncrd = f * crd[:,:2] / crd[:,2,np.newaxis]

		# Interpolate the projected data onto the image grid
		x, y = self.imgcoords(tm, tn)

		pixels = griddata(ncrd, img.reshape((m * n,), order='C'), (x, y))
		depth = griddata(ncrd, crd[:,-1], (x, y))

		return pixels, depth
