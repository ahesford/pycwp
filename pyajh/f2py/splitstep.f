cF77FLAGS(gnu95)=-fopenmp
c Initialize FFTW to use threads, if appropriate
      subroutine initfft(nt)
cf2py integer, optional, intent(in,out) :: nt=0
      implicit none
      include 'fftw3.f'
      integer nt, ierr

!$    integer OMP_GET_MAX_THREADS
!$    IF (nt .LT. 1) THEN
!$      nt = OMP_GET_MAX_THREADS()
!$    ENDIF

      IF (nt .GT. 1) THEN
        call sfftw_init_threads(ierr)
        call sfftw_plan_with_nthreads(nt)
      ENDIF
      end subroutine initfft

c Clean up the FFTW remnants
      subroutine cleanfft(nt)
      include 'fftw3.f'
      integer nt
      IF (nt .GT. 1) THEN
        call sfftw_cleanup_threads
      ELSE
        call sfftw_cleanup
      ENDIF
      end subroutine cleanfft

      subroutine idxmap(i, j, k, m)
cf2py intent(out) :: i, j
      implicit none
      integer i, j, k, m

      i = mod(k - 1, m) + 1
      j = ((k - 1) / m) + 1
      end subroutine idxmap

c Out-of-place (forward or inverse) Fourier transform of an array
      subroutine fftfield(dir, output, input, m, n)
c Arguments:
c     dir: The direction (FFTW_FORWARD or FFTW_INVERSE)
c     fld: The field to transform
c     m,n: The dimensions of the field
cf2py intent(out) :: output
cf2py intent(hide) :: m, n
      implicit none
      include 'fftw3.f'
      integer m,n
      complex input(m,n), output(m,n)
      integer dir

      integer*8 plan
      integer i, j, k, p

      call sfftw_plan_dft_2d(plan, m, n, output, 
     +                       output, dir, FFTW_MEASURE)

      p = m * n
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k)
      do 01 k = 1, p
        call idxmap(i, j, k, m)
01      output(i,j) = input(i,j)
!$OMP END PARALLEL DO

      call sfftw_execute_dft(plan, output, output)
      call sfftw_destroy_plan(plan)
      end subroutine fftfield

c Scale the elements of an array by the product of its dimensions
      subroutine fftscale(fld, m, n)
c Arguments:
c     fld: The array to scale
c     m,n: The dimensions of the field
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
      implicit none
      integer m,n
      complex fld(m,n)

      integer i, j, k, p

      p = m * n
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k)
      do 01 k = 1, p
        call idxmap(i, j, k, m)
01      fld(i,j) = fld(i,j) / p
!$OMP END PARALLEL DO
      end subroutine fftscale

c Propagate, in-place, a spectral field through a homogeneous medium
      subroutine propagate(fld, k0, kx, ky, h, m, n)
c Arguments:
c     fld:   The field to propagte
c     k0:    The reference wave number, unitless
c     kx,ky: The x and y spatial frequencies in FFT order
c     h:     The propagation distance in wavelengths
c     m,n:   The dimensions of the field
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
      implicit none
      integer m,n
      complex fld(m,n)
      real k0, kx(m), ky(n), h

      integer i, j, k, np
      complex kz

      np = m * n

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(kz,i,j,k)
      DO 01 k = 1, np
        call idxmap(i, j, k, m)
        kz = csqrt(cmplx(k0**2 - kx(i)**2 - ky(j)**2))
01      fld(i,j) = fld(i,j) * cexp(cmplx(0., 1.) * kz * h)
!$OMP END PARALLEL DO
      end subroutine propagate

c Apply, in-place, a phase screen to a field
      subroutine phasescreen(fld, eta, k0, h, m, n)
c Arguments:
c     fld: The field requiring correction
c     eta: The index of refraction through which the field is propagated
c     k0:  The reference wave number, unitless
c     h:   The propagation distance in wavelengths
c     m,n: The dimensions of the field and medium
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
      implicit none
      integer m,n
      complex fld(m,n), eta(m,n)
      real k0, h

      integer i, j, k, np
      complex scr

      np = m * n

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,scr)
      DO 01 k = 1, np
        call idxmap(i, j, k, m)
        scr = cexp(cmplx(0., 1.) * k0 * h * (eta(i,j) - 1.))
01      fld(i,j) = scr * fld(i,j)
!$OMP END PARALLEL DO
      end subroutine phasescreen


c Apply, in-place, a wide-angle correction to a propagating field
      subroutine wideangle(fld, lap, eta, k0, h, m, n)
c Arguments:
c     fld: The field requiring correction
c     lap: The (negative) scaled Laplacian of the field
c     eta: The index of refraction through which the field is propagated
c     k0:  The reference wave number, unitless
c     h:   The propagation distance in wavelengths
c     m,n: The dimensions of the field and medium
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
      implicit none
      integer m,n
      complex fld(m,n), eta(m,n), lap(m,n)
      real k0, h

      integer i, j, k, np
      complex cor, e

      np = m * n

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,cor,e)
      DO 01 k = 1, np
        call idxmap(i, j, k, m)
        e = eta(i,j)
        cor = cmplx(0., 1.) * k0 * h * (e - 1.) / (2. * e)
01      fld(i,j) = fld(i,j) + cor * lap(i,j)
!$OMP END PARALLEL DO
      end subroutine wideangle


c Compute the scaled, negative Laplacian of a field.
c On output, an inverse Fourier transform is performed on the field.
      subroutine laplacian(lap, fld, k0, kx, ky, m, n)
c Arguments:
c     lap:   The (negative) Laplacian of the field
c     fld:   The field that will be differentiated
c     k0:    The reference wave number, unitless
c     kx,ky: The x and y spatial frequencies in FFT order
c     m,n:   The dimensions of the field and medium
cf2py intent(in,out) :: lap, fld
cf2py intent(hide) :: m, n
      implicit none
      integer m,n
      complex fld(m,n), lap(m,n)
      real k0, kx(m), ky(n)

      integer i, j, k, np
      real kt

      np = m * n

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,kt)
      DO 01 k = 1, np
        call idxmap(i, j, k, m)
        kt = (kx(i)**2 + ky(j)**2) / k0**2
01      lap(i,j) = fld(i,j) * kt
!$OMP END PARALLEL DO
      end subroutine laplacian

c Convert an object contrast into an index of refraction
      subroutine obj2eta(eta, obj, m, n)
c Arguments:
c     eta: The index of refraction
c     obj: The object contrast
c     m,n: The dimensions of the array
cf2py intent(out) :: eta
cf2py intent(hide) :: m,n
      implicit none
      integer m,n
      complex obj(m,n), eta(m,n)

      integer i, j, k, np

      np = m * n
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k)
      do 01 k = 1, np
        call idxmap(i, j, k, m)
01      eta(i,j) = csqrt(obj(i,j) + 1.)
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
      implicit none
      integer m,n
      complex efrac(m,n), cur(m,n), next(m,n)

      integer i, j, k, np

      np = m * n
!$OMP PARALLEL DO DEFAULT(SHARED) private(i,j,k)
      do 01 k = 1, np
        call idxmap(i, j, k, m)
01      efrac(i,j) = cur(i,j) / next(i,j)
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
      implicit none
      integer m, n, l
      complex fld(m,n)

      integer i, k
      real h, pi
      parameter (pi = 3.141592653589793)

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(k,i,h)
      do 01 k = 1, l
        h = sin(pi * (k - 1) / (2 * l - 1))**2
        do 02 i = 1, m
          fld(i,k) = fld(i,k) * h
02        fld(i,n-k+1) = fld(i,n-k+1) * h
        do 03 i = 1, n
          fld(k,i) = fld(k,i) * h
03        fld(m-k+1,i) = fld(m-k+1,i) * h
01      continue
!$OMP END PARALLEL DO
      end subroutine hann

c Advance the field through a slice of medium
      subroutine advance(fld, eta, k0, kx, ky, h, l, m, n)
c Arguments:
c     fld:   The field to be advanced
c     eta:   The index of refraction of the medium
c     k0:    The reference wave number, unitless
c     kx,ky: The transverse spatial frequencies
c     h:     The propagate distance in wavelengths
c     l:     The width of a Hann window to apply to each boundary
c     m,n:   The dimensions of the slice
cf2py intent(in,out) :: fld
cf2py intent(hide) :: m, n
      implicit none
      include 'fftw3.f'
      integer m,n,l
      complex fld(m,n), eta(m,n)
      real k0, h, kx(m), ky(n)

      complex lap(m,n), buf(m,n)

c Window to attenuate the field near the boundaries
      if (l .GT. 0) then
        call hann(fld, l, m, n)
      endif

c Tranform and propagate the field
      call fftfield(FFTW_FORWARD, buf, fld, m, n)
      call propagate(buf, k0, kx, ky, h, m, n)
c Take the Laplacian of the field
      call laplacian(lap, buf, k0, kx, ky, m, n)
c Inverse transform (and scale) the field and its Laplacian
      call fftfield(FFTW_BACKWARD, fld, buf, m, n)
      call fftscale(fld, m, n)
      call fftfield(FFTW_BACKWARD, buf, lap, m, n)
      call fftscale(buf, m, n)
c Apply wide-angle corrections and the inhomogeneous phase screen
      call wideangle(fld, buf, eta, k0, h, m, n)
      call phasescreen(fld, eta, k0, h, m, n)
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
      implicit none
      integer m, n
      complex fwd(m,n), back(m,n), prev(m,n), efrac(m,n)
      real tau

      integer i, j, k, np
      complex pval, nval, ep1, em1

      np = m * n

!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,k,pval,nval,ep1,em1)
      do 01 k = 1, np
        call idxmap(i, j, k, m)
        ep1 = 1. + efrac(i,j)
        em1 = 1. - efrac(i,j)
        pval = prev(i,j) * (1. - 2. / tau)
        nval = (ep1 * fwd(i,j) + em1 * back(i,j)) / tau
01      fwd(i,j) = pval + nval
!$OMP END PARALLEL DO
      end subroutine update
