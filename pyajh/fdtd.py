import numpy as np

class PML:
	'''
	A class to encapsulate first-order pressure and particle velocity
	updates in a perfectly matched layer where the boundaries are forced,
	either by boundary conditions or exchange with another medium.
	'''

	def __init__(self, c, dim, dt, h, sig, smap):
		'''
		Initialize a PML with a homogeneous soudn speed c. The tuple
		dim gives the 3-dimensional shape of the region, including each
		boundary face. The time step dt and the spatial step h are
		scalars. The list sig should contain the 1-D attenuation
		profile for the boundary layer in one side of the PML. The
		i-element list smap of lists specifies which edges of the PML
		will be attenuated; smap[i][0] == True to attenuate the left
		edge along axis i, and smap[i][1] == True to attenuate the
		right edge along axis i.
		'''

		# Copy the parameters of the PML
		self.csq = c**2
		self.h = h
		self.dt = dt

		# Initialize the pressure components in the PML
		self.px, self.py, self.pz = [np.zeros(dim) for i in range(3)]

		# Initialize the velocity components in the PML
		self.ux = np.zeros((dim[0] - 1, dim[1], dim[2]))
		self.uy = np.zeros((dim[0], dim[1] - 1, dim[2]))
		self.uz = np.zeros((dim[0], dim[1], dim[2] - 1))

		# Initialize the attenuation profiles
		sigma = [np.zeros(dim) for i in range(3)]
		# Append a zero value to the inside of the sigma profile
		sig = list(sig) + [0.]

		# Compute the indices starting from the left edges
		left = np.mgrid[:dim[0], :dim[1], :dim[2]]
		# Compute the indices starting from the right edges
		right = np.mgrid[dim[0]-1:-1:-1, dim[1]-1:-1:-1, dim[2]-1:-1:-1]

		# Add in the PML profiles as specified in the map
		for s, m, cl, cr in zip(sigma, smap, left, right):
			if m[0]: s += cl.choose(sig, mode='clip')
			if m[1]: s += cr.choose(sig, mode='clip')

		# Compute the scaling factors for pressure
		self.txp, self.typ, self.tzp = [[1. / dt - 0.5 * s,
			1. / dt + 0.5 * s] for s in sigma]

		# The velocity factors are averaged from the pressure factors
		self.txu = [0.5 * (t[1:,:,:] + t[:-1,:,:]) for t in self.txp]
		self.tyu = [0.5 * (t[:,1:,:] + t[:,:-1,:]) for t in self.typ]
		self.tzu = [0.5 * (t[:,:,1:] + t[:,:,:-1]) for t in self.tzp]


	def pressure(self):
		'''
		Return the sum of the three pressure components in the PML. If
		left is true, return the left PML pressure; otherwise, return
		the right PML pressure.
		'''

		return self.px + self.py + self.pz


	def update(self):
		'''
		Update the particle velocity everywhere in the PML and the
		pressure everywhere away from each boundary. The boundaries are
		all forced separately.
		'''

		# Grab the total pressure throughout the PML
		p = self.pressure()

		# Scale the previous velocity and pressure components
		self.ux *= self.txu[0] / self.txu[1]
		self.uy *= self.tyu[0] / self.tyu[1]
		self.uz *= self.tzu[0] / self.tzu[1]

		self.px *= self.txp[0] / self.txp[1]
		self.py *= self.typ[0] / self.typ[1]
		self.pz *= self.tzp[0] / self.tzp[1]

		# Add pressure derivatives to the velocity components
		self.ux -= (p[1:,:,:] - p[:-1,:,:]) / (self.txu[1] * self.h)
		self.uy -= (p[:,1:,:] - p[:,:-1,:]) / (self.tyu[1] * self.h)
		self.uz -= (p[:,:,1:] - p[:,:,:-1]) / (self.tzu[1] * self.h)

		# Add velocity derivatives to the pressure components
		self.px[1:-1,:,:] -= self.csq * (self.ux[1:,:,:] 
				- self.ux[:-1,:,:]) / (self.txp[1][1:-1,:,:] * self.h)
		self.py[:,1:-1,:] -= self.csq * (self.uy[:,1:,:] 
				- self.uy[:,:-1,:]) / (self.typ[1][:,1:-1,:] * self.h)
		self.pz[:,:,1:-1] -= self.csq * (self.uz[:,:,1:] 
				- self.uz[:,:,:-1]) / (self.tzp[1][:,:,1:-1] * self.h)

	
	def boundary(self, xl = 0., xr = 0., yl = 0., yr = 0., zl = 0., zr = 0.):
		'''
		Enforce boundary conditions at the left (0-index) and right
		(-1-index) boundaries in each plane. The boundary value is
		divided by three to be equally distributed among each of the
		three pressure components.
		'''

		# Enforce the left and right x-boundaries for each component
		self.px[0,:,:] = self.py[0,:,:] = self.pz[0,:,:] = xl / 3.
		self.px[-1,:,:] = self.py[-1,:,:] = self.pz[-1,:,:] = xr / 3.

		# Enforce the left and right y-boundaries for each component
		self.px[:,0,:] = self.py[:,0,:] = self.pz[:,0,:] = yl / 3.
		self.px[:,-1,:] = self.py[:,-1,:] = self.pz[:,-1,:] = yr / 3.

		# Enforce the left and right z-boundaries for each component
		self.px[:,:,0] = self.py[:,:,0] = self.pz[:,:,0] = zl / 3.
		self.px[:,:,-1] = self.py[:,:,-1] = self.pz[:,:,-1] = zr / 3.



class Helmholtz:
	'''
	A class to encapsulate second-order pressure updates in a bounded
	medium. The boundaries are independently forced directly or through
	exchanges with other media.
	'''

	def __init__(self, c, dt, h, srcfunc):
		'''
		Initialize the sound-speed c, time step dt and spatial step h.
		The coroutine srcfunc should provide a time-dependent function
		that describes the source over the same grid as the sound
		speed.
		'''

		# Make a copy of the sound-speed map and parameters
		self.c = c.copy()
		self.dt = dt
		self.h = h

		# The source is a generator created by a coroutine
		self.source = srcfunc()

		# The pre-computed scale factors make updates more efficient
		self.rsq = (self.c * self.dt / self.h)**2
		self.csq = self.c**2

		# Initialize the pressure at three initial time steps
		# The current time step is p[0]
		# The next time step is p[1]
		# The previous time step is p[-1] (in this case, p[2])
		self.p = [np.zeros_like(self.c) for i in range(3)]


	def update(self):
		'''
		Update the pressure everywhere away from the boundary. The
		boundaries are forced separately.
		'''

		# Perform the previous-time updates
		self.p[1] = ((2. - 6. * self.rsq) * self.p[0] - self.p[-1]
				- self.csq * self.dt**2 * self.source.next())

		# Perform the current-time, neighboring updates
		self.p[1][1:-1,:,:] += self.rsq[1:-1,:,:] * (self.p[0][2:,:,:]
				+ self.p[0][:-2,:,:])
		self.p[1][:,1:-1,:] += self.rsq[:,1:-1,:] * (self.p[0][:,2:,:]
				+ self.p[0][:,:-2,:])
		self.p[1][:,:,1:-1] += self.rsq[:,:,1:-1] * (self.p[0][:,:,2:]
				+ self.p[0][:,:,:-2])

		# Mark the current time step as the previous
		self.p[-1] = self.p[0]

		# Mark the next time step as the current
		self.p[0] = self.p[1]


	def boundary(self, xl = 0., xr = 0., yl = 0., yr = 0., zl = 0., zr = 0.):
		'''
		Enforce boundary conditions at the left (0-index) and right
		(-1-index) boundaries in each plane.
		'''

		# Enforce the left and right x-boundaries for each component
		self.p[0][0,:,:] = xl
		self.p[0][-1,:,:] = xr

		# Enforce the left and right y-boundaries for each component
		self.p[0][:,0,:] = yl
		self.p[0][:,-1,:] = yr

		# Enforce the left and right z-boundaries for each component
		self.p[0][:,:,0] = zl
		self.p[0][:,:,-1] = zr



class FDTD:
	'''
	A simple FDTD engine that uses a hybrid scalar/vector formulation to
	efficiently compute the propagation of acoustic waves in a medium
	bounded by a perfectly matched layer.
	'''

	def __init__(self, c, cbg, dt, h, sigma, srcfunc):
		'''
		Initialize the Helmholtz and first-order solvers with variable
		sound-speed grid c, background PML sound speed cbg, time step
		dt, spatial step h, and a 1-D profile sigma describing
		attenuation from the outer edge to the inner edge of the PML.
		The coroutine srcfunc specifies the source term over the same
		grid as the sound speed.
		'''

		# Force boundary overlaps to have background sound speed
		c[:2,:,:] = cbg
		c[-2:,:,:] = cbg
		c[:,:2,:] = cbg
		c[:,-2:,:] = cbg
		c[:,:,:2] = cbg
		c[:,:,-2:] = cbg

		# Initialize the Helmholtz region
		# Note that the boundary overlaps the PML
		self.helmholtz = Helmholtz(c, dt, h, srcfunc)

		# Copy the PML thickness
		self.l = len(sigma)

		# Note the size of the total grid, excluding overlap
		self.tsize = tuple(d + 2 * (self.l - 1) for d in c.shape)
		# Note the size of the Helmholtz grid
		self.hsize = c.shape[:]

		tg, hg, lp = self.tsize, self.hsize, self.l + 1

		# The shapes of the PMLs along each axis
		shapes = [list(hg[:i]) + [lp] + list(tg[i+1:]) for i in range(3)]

		# Placeholders for the PML attenuation maps
		left, right = [True, False], [False, True]
		tedges, fedges = [[True]*2]*3, [[False]*2]*3

		# Note which edges will be attenuated using PMLs
		# Don't attenuate edges shared with other regions
		amaps = [[fedges[:i] + [s] + tedges[i+1:]
			for s in [left, right]] for i in range(3)]

		# The PML list contains three lists, holding, respectively, the
		# left and right sides of the x, y and z PMLs
		self.pml = [[PML(cbg, dim, dt, h, sigma, a) for a in aax]
				for dim, aax in zip(shapes, amaps)]


	def exchange(self):
		'''
		Exchange the boundary values between the PML and Helmholtz regions.
		'''

		# Shorthand for PML thickness offset indices
		l = self.l
		lm = self.l - 1
		lp = self.l + 1

		# Create array to hold total pressure value
		p = self.pressure()

		# Set the boundary values for each PML

		# The x-axis PMLs: interior x surface set from Helmholtz
		self.pml[0][0].boundary(xr = p[l,:,:])
		self.pml[0][1].boundary(xl = p[-lp,:,:])

		# The y-axis PMLs: interior y surface set from Helmholtz
		# The x surfaces are set from the x-axis PMLs
		self.pml[1][0].boundary(xl = p[lm, :lp, :], xr = p[-l, :lp, :],
				yr = p[lm:-lm, l, :])
		self.pml[1][1].boundary(xl = p[lm, -lp:, :], xr = p[-l, -lp:, :],
				yl = p[lm:-lm, -lp, :])

		# The z-axis PMLs: interior z surface set from Helmholtz
		# The x and y surfaces are set from the x- and y-axis PMLs
		self.pml[2][0].boundary(
				xl = p[lm, lm:-lm, :lp],  xr = p[-l, lm:-lm, :lp], 
				yl = p[lm:-lm, lm, :lp],  yr = p[lm:-lm, -l, :lp], 
				zr = p[lm:-lm, lm:-lm, l])
		self.pml[2][1].boundary(
				xl = p[lm, lm:-lm, -lp:],  xr = p[-l, lm:-lm, -lp:], 
				yl = p[lm:-lm, lm, -lp:],  yr = p[lm:-lm, -l, -lp:], 
				zl = p[lm:-lm, lm:-lm, -lp])

		# Copy the PML pressures to the Helmholtz boundary
		self.helmholtz.boundary (
				p[lm, lm:-lm, lm:-lm], p[-l, lm:-lm, lm:-lm],
				p[lm:-lm, lm, lm:-lm], p[lm:-lm, -l, lm:-lm],
				p[lm:-lm, lm:-lm, lm], p[lm:-lm, lm:-lm, -l])

		# Return the total pressure
		return p


	def pressure(self):
		'''
		Return an array containing the total pressure over the union of
		the Helmholtz and PML grids.
		'''

		# Shorthand for PML thickness offset
		l = self.l

		# Create array to hold total pressure value
		p = np.zeros(self.tsize)

		# Fill the Helmholtz pressure in the center
		p[l:-l,l:-l,l:-l] = self.helmholtz.p[0][1:-1,1:-1,1:-1]
		# Copy the PML pressure from the left and right x edges
		p[:l,:,:] = self.pml[0][0].pressure()[:l,:,:]
		p[-l:,:,:] = self.pml[0][1].pressure()[-l:,:,:]
		# Copy the PML pressure from the left and right y edges
		p[l:-l,:l,:] = self.pml[1][0].pressure()[1:-1,:l,:]
		p[l:-l,-l:,:] = self.pml[1][1].pressure()[1:-1,-l:,:]
		# Copy the PML pressure from the left and right z edges
		p[l:-l,l:-l,:l] = self.pml[2][0].pressure()[1:-1,1:-1,:l]
		p[l:-l,l:-l,-l:] = self.pml[2][1].pressure()[1:-1,1:-1,-l:]
		
		return p


	def update(self):
		'''
		Update the Helmholtz and PML fields.
		'''

		# Update the Helmholtz pressure away from the forced boundaries
		self.helmholtz.update()

		# Update the PML fields away from the forced boundaries
		for pleft, pright in self.pml:
			pleft.update()
			pright.update()

		# Return the pressure after exchanging boundary values
		return self.exchange()
