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


c Propagate, in place, a spectral field through a homogeneous medium
      subroutine propagate(fld, k0, h, dz, m, n)
c Arguments:
c     fld: The field to propagte
c     k0:  The reference wave number, unitless
c     h:   The transverse spatial sampling interval in wavelengths
c     dz:  The propagation distance in wavelengths
c     m,n: The dimensions of the field
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py threadsafe
      use fft, only : fftexec
      implicit none
      include 'fftw3.f'
      integer m,n
      complex fld(m,n)
      real k0, h, dz

      integer i, j
      complex kz
      real kx, ky, fftfreq, p

      p = real(m * n)

      call fftexec(FFTW_FORWARD, fld, m, n)

c Scale the propagation by (m * n) to counter FFT scaling
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(kx,ky,kz,i,j)
      do j = 1, n
        ky = fftfreq(j, n, h)
        do i = 1, m
          kx = fftfreq(i, m, h)
          kz = -k0 + csqrt(cmplx(k0**2 - kx**2 - ky**2))
          fld(i,j) = fld(i,j) * cexp(cmplx(0., 1.) * dz * kz) / p
        enddo
      enddo
!$OMP END PARALLEL DO

      call fftexec(FFTW_BACKWARD, fld, m, n)
      end subroutine propagate


c Apply, in place, a phase screen to a field
      subroutine phasescreen(fld, eta, k0, dz, m, n)
c Arguments:
c     fld: The field requiring correction
c     eta: The index of refraction through which the field is propagated
c     k0:  The reference wave number, unitless
c     dz:  The propagation distance in wavelengths
c     m,n: The dimensions of the field and medium
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      integer m,n
      complex fld(m,n), eta(m,n)
      real k0, dz

      integer i, j
      complex scr

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,scr)
      do j = 1, n
        do i = 1, m
          scr = cexp(cmplx(0., 1.) * k0 * dz * eta(i,j))
          fld(i,j) = scr * fld(i,j)
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine phasescreen

c Apply, in place, wide-angle corrections according to Lin and Duda 2012
      subroutine wideangle(fld, eta, k0, h, dz, m, n, maxit, tol)
c Arguments:
c     fld: The field to be corrected
c     eta: The index of refraction through which the field is propagated
c     k0:  The reference wave number, unitless
c     h:   The transverse spatial sampling interval in wavelengths
c     dz:  The propagation distance in wavelengths
c     m,n: The dimensions of the field and medium
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py real, optional :: tol = 1e-6
cf2py integer, optional :: maxit = 10
cf2py threadsafe
      use fft, only : fftexec
      implicit none
      include 'fftw3.f'
      integer m, n, maxit
      complex fld(m,n), eta(m,n)
      real k0, h, dz, tol

      complex delta, kz, u(m,n)
      integer i, j, l
      real nnum, nden, kx, ky, fftfreq, p

      delta = cmplx(0., -1.) * k0 * dz
      p = real(m * n)

c Apply the spatial operator N = (n - 1) to the field and store in u
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j)
      do j = 1, n
        do i = 1, m
          u(i,j) = (eta(i,j) - 1) * fld(i,j)
        enddo
      enddo
!$OMP END PARALLEL DO

      do l = 1, maxit
c Spectrally evaluate the operation Lu
c Scale the result by (m * n) to counter FFT scaling
        call fftexec(FFTW_FORWARD, u, m, n)
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,kx,ky,kz)
        do j = 1, n
          ky = fftfreq(j, n, h) / k0
          do i = 1, m
            kx = fftfreq(i, m, h) / k0
            kz = (-1 + csqrt(cmplx(1 - kx**2 - ky**2))) / p
            u(i,j) = kz * u(i,j)
          enddo
        enddo
!$OMP END PARALLEL DO
        call fftexec(FFTW_BACKWARD, u, m, n)

        nnum = 0.
        nden = 0.

c Now finish computing the update u = -i k0 dz * LNu / m
c Add the new update to the total field
c Also apply the spatial operator N to the update for the next round
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j) REDUCTION(+:nnum,nden)
        do j = 1, n
          do i = 1, m
            u(i,j) = delta * u(i,j) / l
            fld(i,j) = fld(i,j) + u(i,j)
            nnum = nnum + cabs(u(i,j))
            u(i,j) = (eta(i,j) - 1) * u(i,j)
            nden = nden + cabs(fld(i,j))
          enddo
        enddo
!$OMP END PARALLEL DO

        if ((nden .LT. tol .AND. nnum .LT. tol)
     +      .OR. nnum / nden .LT. TOL) exit
        enddo
      end subroutine wideangle


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


c Compute the ratio of the current index of refraction to the next one
        subroutine etafrac(efrac, cur, next, m, n)
c Arguments:
c     efrac: The ratio of refractive indices
c     cur:   The refractive index in the current slab
c     next:  The refractive index in the next slab
c     m,n:   The dimensions of the arrays
cf2py intent(out) :: efrac
cf2py intent(hide) :: m,n
cf2py threadsafe
      implicit none
      integer m,n
      complex efrac(m,n), cur(m,n), next(m,n)

      integer i, j

!$OMP PARALLEL DO DEFAULT(SHARED) private(i,j)
      do j = 1, n
        do i = 1, m
          efrac(i,j) = cur(i,j) / next(i,j)
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine etafrac


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


c Advance the field through a slice of medium
      subroutine advance(fld, eta, k0, h, dz, l, m, n)
c Arguments:
c     fld: The field to be advanced
c     eta: The index of refraction of the medium
c     k0:  The reference wave number, unitless
c     h:   The transverse sampling interval in wavelengths
c     dz:  The propagation distance in wavelengths
c     l:   The width of a Hann window to apply to each boundary
c     m,n: The dimensions of the slice
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
cf2py threadsafe
      implicit none
      include 'fftw3.f'
      integer m,n,l
      complex fld(m,n), eta(m,n)
      real k0, h, dz

c Window to attenuate the field near the boundaries
      if (l .GT. 0) then
        call hann(fld, l, m, n)
      endif

c Tranform and propagate the field
      call propagate(fld, k0, h, dz, m, n)
c Apply wide-angle corrections and the inhomogeneous phase screen
      call wideangle(fld, eta, k0, h, dz, m, n, 1, 1e-6)
      call phasescreen(fld, eta, k0, dz, m, n)
      end subroutine advance


c Apply a relaxation update to the field propagated through a slab
      subroutine update(fwd, back, prev, efrac, tau, m, n)
c Arguments:
c     fwd:   The forward-propagating field to be udpated
c     back:  The field proapgting counter to the field
c     prev:  The prior guess of the forward-traveling field
c     efrac: The ratio of the current to the next refractive indices
c     tau:   The relaxation paramter
c     m,n:   The dimensions of the field
cf2py intent(in,out) :: fwd
cf2py intent(hide) :: m,n
cf2py optional :: tau=2.
cf2py threadsafe
      implicit none
      integer m, n
      complex fwd(m,n), back(m,n), prev(m,n), efrac(m,n)
      real tau

      integer i, j
      complex pval, nval, ep1, em1

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,pval,nval,ep1,em1)
      do j = 1, n
        do i = 1, m
          ep1 = 1. + efrac(i,j)
          em1 = 1. - efrac(i,j)
          pval = prev(i,j) * (1. - 2. / tau)
          nval = (ep1 * fwd(i,j) + em1 * back(i,j)) / tau
          fwd(i,j) = pval + nval
        enddo
      enddo
!$OMP END PARALLEL DO
      end subroutine update
