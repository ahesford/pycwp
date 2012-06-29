cF77FLAGS(gnu95)=-fopenmp
c Compute the action of the split-step Padé operator
c 
c     Q = a + b * obj + b * (d_{xx} + d_{yy}) / (k_0**2)
c
c acting on a field fld, storing the result in y.
c
c Dirichlet boundary conditions are assumed
      subroutine scatop(obj, fld, h, k0, a, b, y, n, m)
c Arguments:
c     obj: The object contrast of the medium
c     fld: The field to which the operator Q is applied
c     h:   The spatial step for finite-difference approximations to Q
c     k0:  The reference wave number
c     a,b: The coefficients a and b in the split-step Padé operator
c     y:   The solution y = Q * fld
c     n,m: The dimensions of obj, fld and y
cf2py intent(out) :: y
      integer n, m
      complex*16 obj(n,m), fld(n,m), y(n,m), a, b
      real*8 h, k0

      complex*16 dxm, dfd, df

      integer i, j, nm1, mm1

      integer omp_get_max_threads

!$    PRINT *, 'USING THREADS:', omp_get_max_threads()

      nm1 = n - 1
      mm1 = m - 1

      dxm = b / (k0 * h)**2
      dfd = a - 4. * dxm

c Compute terms away from the boundaries
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,j,df)
      do 10 i = 2, nm1
      do 10 j = 2, mm1
        df = fld(i+1,j) + fld(i-1,j) + fld(i,j+1) + fld(i,j-1)
10      y(i,j) = (dfd + b * obj(i,j)) * fld(i,j) + dxm * df
!$OMP END PARALLEL DO

c Compute derivatives along the x boundaries
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(j,df)
      do 20 j = 2, mm1
        df = fld(2,j) + fld(1,j-1) + fld(1,j+1)
        y(1,j) = (dfd + b * obj(1,j)) * fld(1,j) + dxm * df
        df = fld(nm1,j) + fld(n,j-1) + fld(n,j+1)
20      y(n,j) = (dfd + b * obj(n,j)) * fld(n,j) + dxm * df
!$OMP END PARALLEL DO

c Compute the derivatives along the y boundaries
!$OMP PARALLEL DO DEFAULT(SHARED) PRIVATE(i,df)
      do 30 i = 2, nm1
        df = fld(i+1,1) + fld(i-1,1) + fld(i,2)
        y(i,1) = (dfd + b * obj(i,1)) * fld(i,1) + dxm * df
        df = fld(i+1,m) + fld(i-1,m) + fld(i,mm1)
30      y(i,m) = (dfd + b * obj(i,m)) * fld(i,m) + dxm * df
!$OMP END PARALLEL DO

c Compute the terms at each of the four corners
!$OMP PARALLEL WORKSHARE DEFAULT(SHARED)
      y(1,1) = (dfd + b * obj(1,1)) * fld(1,1) + 
     +         dxm * (fld(1,2) + fld(2,1))
      y(1,m) = (dfd + b * obj(1,m)) * fld(1,m) + 
     +         dxm * (fld(1,m-1) + fld(2,m))
      y(n,1) = (dfd + b * obj(n,1)) * fld(n,1) + 
     +         dxm * (fld(n-1,1) + fld(n,2))
      y(n,m) = (dfd + b * obj(n,m)) * fld(n,m) +
     +         dxm * (fld(n-1,m) + fld(n,m-1))
!$OMP END PARALLEL WORKSHARE
      end subroutine scatop
