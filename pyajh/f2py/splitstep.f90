c A module to encapsulate forward and inverse FFTW transforms
      module fft
        implicit none
c The number of threads used for the FFT
        integer :: nthread

        contains

c Initialize the FFTW library
        subroutine init(nt)
cf2py integer, optional :: nt=0
          implicit none
          integer nt, ierr
c Determine, if necessary, the default number of threads using OpenMP
!$        integer OMP_GET_MAX_THREADS
!$        if (nt .LT. 1) then
!$          nt = OMP_GET_MAX_THREADS()
!$        endif

c Use at least one thread
          nthread = max(nt, 1)
c Attempt to initialize FFTW to use multiple threads if desired
          if (nthread .GT. 1) then
            call sfftw_init_threads(ierr)
            call sfftw_plan_with_nthreads(nthread)
            if (ierr .EQ. 0) then
              call sfftw_cleanup_threads
              nthread = 1
            endif
          endif
        end subroutine init


c Clean up the FFTW library before exit
        subroutine cleanup
          if (nthread .GT. 1) then
            call sfftw_cleanup_threads
          else
            call sfftw_cleanup
          endif
        end subroutine cleanup


c Plan forward and inverse in-place, (m,n)-point DFTs
c The plans are discarded, but knowledge should remain available
c for use in future plans that use FFTW_ESTIMATE
        subroutine plan(m, n)
          implicit none
          include 'fftw3.f'
          integer m, n
          complex arr(m, n)

          integer*8 fplan, bplan

          call sfftw_plan_dft_2d(fplan, m, n, arr, arr,
     +                           FFTW_FORWARD, FFTW_MEASURE)
          call sfftw_plan_dft_2d(bplan, m, n, arr, arr,
     +                           FFTW_BACKWARD, FFTW_MEASURE)
          call sfftw_destroy_plan(fplan)
          call sfftw_destroy_plan(bplan)
        end subroutine plan


c Perform an in-place FFT using FFTW
c Note that inverse transforms are unscaled
        subroutine fftexec(dir, arr, m, n)
cf2py intent(in,out) :: arr
cf2py intent(hide) :: m, n
cf2py threadsafe
          implicit none
          include 'fftw3.f'
          integer m, n, dir
          complex arr(m, n)

          integer*8 plan

c The estimate should reuse knowledge from prior calls to plan(m,n)
c Estimation is necessary to avoid clobbering array contents
          call sfftw_plan_dft_2d(plan, m, n, arr,
     +                           arr, dir, FFTW_ESTIMATE)
          call sfftw_execute_dft(plan, arr, arr)
          call sfftw_destroy_plan(plan)
        end subroutine fftexec
      end module fft


c Compute the i-th (one-indexed) DFT frequency bin
c for an m-point FFT with sample spacing h
      function fftfreq(i, m, h)
        implicit none
        real fftfreq, h
        integer i, m

        integer half
        real pi
        parameter (pi = 3.141592653589793)

        half = (m - 1) / 2
        if (i .LE. half) then
          fftfreq = real(i - 1)
        else
          fftfreq = real(i - 1 - m)
        endif
        fftfreq = fftfreq * 2. * pi / (real(m) * h)
      end function fftfreq


c Convert an object contrast into an index of refraction
      subroutine obj2eta(eta, obj, m, n)
c Arguments:
c     eta: The index of refraction
c     obj: The object contrast
c     m,n: The dimensions of the array
cf2py intent(out) :: eta
cf2py intent(hide) :: m,n
cf2py threadsafe
      implicit none
      integer m,n
      complex obj(m,n), eta(m,n)

      integer i, j

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          eta(i,j) = csqrt(obj(i,j) + 1.)
        enddo
      enddo
!$OMP END PARALLEL DO
        end subroutine obj2eta


c Compute the reflection coefficient for a slab interface
        subroutine rcoeff(rc, cur, next, m, n)
c Arguments:
c     rc:   The ratio of refractive indices
c     cur:  The refractive index in the current slab
c     next: The refractive index in the next slab
c     m,n:  The dimensions of the arrays
cf2py intent(out) :: rc
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      integer m, n
      complex rc(m,n), cur(m,n), next(m,n)

      integer i, j

!$OMP PARALLEL DO DEFAULT(SHARED) private(i,j)
      do j = 1, n
        do i = 1, m
          rc(i,j) = 0.5 * (1. - cur(i,j) / next(i,j))
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine rcoeff


c Apply a Hann window to the boundaries of a field.
      subroutine hann(fld, l, m, n)
c Arguments:
c     fld: The field to be windowed
c     l:   The width of the Hann window along each border
c     m,n: The dimensions of the field
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      integer m, n, l
      complex fld(m,n)

      integer i, k
      real h, pi
      parameter (pi = 3.141592653589793)

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(k,i,h)
      do k = 1, l
        h = sin(pi * (k - 1) / (2 * l - 1))**2
        do i = 1, m
          fld(i,k) = fld(i,k) * h
          fld(i,n-k+1) = fld(i,n-k+1) * h
        enddo

        do i = 1, n
          fld(k,i) = fld(k,i) * h
          fld(m-k+1,i) = fld(m-k+1,i) * h
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine hann


c Apply JPA's wide-angle spectral operator to a spectral field
      subroutine specop(ofld, ifld, k0, h, m, n)
c Arguments:
c     ofld: The output field (spectral)
c     ifld: The input field (spectral)
c     k0:   The reference wave number, unitless
c     h:    The transverse sampling interval in wavelengths
c     m,n:  The dimensions of the slice
cf2py intent(out) :: ofld
cf2py intent(hide) :: m, n
cf2py threadsafe
      use fft, only : fftexec
      implicit none
      include 'fftw3.f'
      integer m, n
      complex ifld(m,n), ofld(m,n)
      real k0, h

      integer i, j
      real kx, ky, kt, fftfreq
      complex kz

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,kx,ky,kt,kz)
      do j = 1, n
        ky = fftfreq(j, n, h) / k0
        do i = 1, m
          kx = fftfreq(i, m, h) / k0
          kt = kx**2 + ky**2
          kz = 1 - 0.5 * kt - csqrt(cmplx(1 - kt))
c Divide the spatial operator by kt if it is nonzero
c Otherwise, the numerator will vanish faster to avoid a singularity
          if (i. ne. 1 .or. j .ne. 1) kz = kz / kt
          ofld(i,j) = ifld(i,j) * kz
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine specop


c Advance the field through a slice of medium
      subroutine advance(fld, eta, k0, h, dz, m, n)
c Arguments:
c     fld: The field to be advanced
c     eta: The index of refraction of the medium
c     k0:  The reference wave number, unitless
c     h:   The transverse sampling interval in wavelengths
c     dz:  The propagation distance in wavelengths
c     m,n: The dimensions of the slice
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py threadsafe
      use fft, only : fftexec
      implicit none
      include 'fftw3.f'
      integer m, n
      complex fld(m,n), eta(m,n)
      real k0, h, dz

      integer i, j
      real kx, ky, kt, fftfreq, mn
      complex kz, delta, u(m,n), v(m,n)

      real hsq
      parameter (hsq = 0.1024)

      mn = real(m * n)
      delta = cmplx(0, k0 * dz)

c Transform the field
      call fftexec(FFTW_FORWARD, fld, m, n)

c Apply a spectral wide-angle operators to the field and store in u
      call specop(u, fld, k0, h, m, n)

c Add the wide-angle correction to the field itself and store in v
c Scale u and v by 1 / (m * n) to counter FFT scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          v(i,j) = (fld(i,j) + u(i,j) / hsq) / mn
          u(i,j) = u(i,j) / mn
        enddo
      enddo
!$OMP END PARALLEL DO

c Apply the spatial operator Q to v and u
      call fftexec(FFTW_BACKWARD, u, m, n)
      call fftexec(FFTW_BACKWARD, v, m, n)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,kz)
      do j = 1, n
        do i = 1, m
          kz = eta(i,j)**2 - 1
          v(i,j) = v(i,j) * kz
          u(i,j) = u(i,j) * kz
        enddo
      enddo
!$OMP END PARALLEL DO
      call fftexec(FFTW_FORWARD, u, m, n)
      call fftexec(FFTW_FORWARD, v, m, n)

c Apply the wide-angle spectral operator to v
      call specop(v, v, k0, h, m, n)

c Use the spectral propagator to advance the total field
c Scale the result by 1 / (m * n) to counter FFT scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,kx,ky,kt,kz)
      do j = 1, n
        ky = fftfreq(j, n, h) / k0
        do i = 1, m
          kx = fftfreq(i, m, h) / k0
          kt = kx**2 + ky**2
          kz = csqrt(cmplx(1 - kt))
          fld(i,j) = fld(i,j) + delta * (u(i,j) + v(i,j))
          fld(i,j) = fld(i,j) * cexp(delta * kz) / mn
        enddo
      enddo
!$OMP END PARALLEL DO

c Transform the field back to the spatial domain
      call fftexec(FFTW_BACKWARD, fld, m, n)

c Apply the spatial phase screen to the field
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          fld(i,j) = fld(i,j) * cexp(delta * (eta(i,j) - 1.))
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine advance


c Transmit a forward-propagating field through a slab
c characterized by reflection coefficients rc
      subroutine transmit(fwd, rc, m, n)
c Arguments:
c     fwd: The forward-propagating field to be udpated
c     rc:  The reflection coefficients of the interface
c     m,n: The dimensions of the field
cf2py intent(in,out) :: fwd
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      integer m, n
      complex fwd(m,n), rc(m,n)

      integer i, j

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          fwd(i,j) = (1 - rc(i,j)) * fwd(i,j)
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine transmit


c Compute the reflection of the field bck by an interface
c characterized by reflection coefficients rc and add it
c to the transmission of the field fwd across the interface
      subroutine txreflect(fwd, bck, rc, m, n)
c Arguments:
c     fwd: The forward-traveling field to be transmitted
c     bck: The backward-traveling field to be reflected
c     rc:  The reflection coefficients of the interface
c     tau: The relaxation paramter
c     m,n: The dimensions of the field
cf2py intent(in,out) :: fwd
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      integer m, n
      complex fwd(m,n), bck(m,n), rc(m,n)

      integer i, j

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          fwd(i,j) = (1 - rc(i,j)) * fwd(i,j) + rc(i,j) * bck(i,j)
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine txreflect
