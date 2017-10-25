'''
Ports of Scipy (v. 0.19.0) iterative linear solvers that allow specification of
custom vector norm functions to facilitate parallel computation.
'''

# Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
# Restrictions are listed in the LICENSE file distributed with this package.



__all__ = [ 'lsqr', 'lsmr' ]

import numpy as np
from math import sqrt

from numpy import zeros, infty, atleast_1d
from numpy.linalg import norm

from scipy.sparse.linalg.interface import aslinearoperator

eps = np.finfo(np.float64).eps

def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(A, b, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
         iter_lim=None, show=False, calc_var=False, unorm=None, vnorm=None):
    """Find the least-squares solution to a large, sparse, linear system
    of equations.

    The function solves ``Ax = b``  or  ``min ||b - Ax||^2`` or
    ``min ||Ax - b||^2 + d^2 ||x||^2``.

    The matrix A may be square or rectangular (over-determined or
    under-determined), and may have any rank.

    ::

      1. Unsymmetric equations --    solve  A*x = b

      2. Linear least squares  --    solve  A*x = b
                                     in the least-squares sense

      3. Damped least squares  --    solve  (   A    )*x = ( b )
                                            ( damp*I )     ( 0 )
                                     in the least-squares sense

    Parameters
    ----------
    A : {sparse matrix, ndarray, LinearOperator}
        Representation of an m-by-n matrix.  It is required that
        the linear operator can produce ``Ax`` and ``A^T x``.
    b : array_like, shape (m,)
        Right-hand side vector ``b``.
    damp : float
        Damping coefficient.
    atol, btol : float, optional
        Stopping tolerances. If both are 1.0e-9 (say), the final
        residual norm should be accurate to about 9 digits.  (The
        final x will usually have fewer correct digits, depending on
        cond(A) and the size of damp.)
    conlim : float, optional
        Another stopping tolerance.  lsqr terminates if an estimate of
        ``cond(A)`` exceeds `conlim`.  For compatible systems ``Ax =
        b``, `conlim` could be as large as 1.0e+12 (say).  For
        least-squares problems, conlim should be less than 1.0e+8.
        Maximum precision can be obtained by setting ``atol = btol =
        conlim = zero``, but the number of iterations may then be
        excessive.
    iter_lim : int, optional
        Explicit limitation on number of iterations (for safety).
    show : bool, optional
        Display an iteration log.
    calc_var : bool, optional
        Whether to estimate diagonals of ``(A'A + damp^2*I)^{-1}``.
    unorm, vnorm : callable f(v), optional
        Functions to compute 2-norms in the range of A (unorm) or in the
        solution space (vnorm). Each function should take a single vector
        argument and return its 2-norm as a floating-point value. If either is
        not provided, numpy.linalg.norm will be used by default.

    Returns
    -------
    x : ndarray of float
        The final solution.
    istop : int
        Gives the reason for termination.
        1 means x is an approximate solution to Ax = b.
        2 means x approximately solves the least-squares problem.
    itn : int
        Iteration number upon termination.
    r1norm : float
        ``norm(r)``, where ``r = b - Ax``.
    r2norm : float
        ``sqrt( norm(r)^2  +  damp^2 * norm(x)^2 )``.  Equal to `r1norm` if
        ``damp == 0``.
    anorm : float
        Estimate of Frobenius norm of ``Abar = [[A]; [damp*I]]``.
    acond : float
        Estimate of ``cond(Abar)``.
    arnorm : float
        Estimate of ``norm(A'*r - damp^2*x)``.
    xnorm : float
        ``norm(x)``
    var : ndarray of float
        If ``calc_var`` is True, estimates all diagonals of
        ``(A'A)^{-1}`` (if ``damp == 0``) or more generally ``(A'A +
        damp^2*I)^{-1}``.  This is well defined if A has full column
        rank or ``damp > 0``.  (Not sure what var means if ``rank(A)
        < n`` and ``damp = 0.``)

    Notes
    -----
    LSQR uses an iterative method to approximate the solution.  The
    number of iterations required to reach a certain accuracy depends
    strongly on the scaling of the problem.  Poor scaling of the rows
    or columns of A should therefore be avoided where possible.

    For example, in problem 1 the solution is unaltered by
    row-scaling.  If a row of A is very small or large compared to
    the other rows of A, the corresponding row of ( A  b ) should be
    scaled up or down.

    In problems 1 and 2, the solution x is easily recovered
    following column-scaling.  Unless better information is known,
    the nonzero columns of A should be scaled so that they all have
    the same Euclidean norm (e.g., 1.0).

    In problem 3, there is no freedom to re-scale if damp is
    nonzero.  However, the value of damp should be assigned only
    after attention has been paid to the scaling of A.

    The parameter damp is intended to help regularize
    ill-conditioned systems, by preventing the true solution from
    being very large.  Another aid to regularization is provided by
    the parameter acond, which may be used to terminate iterations
    before the computed solution becomes very large.

    If some initial estimate ``x0`` is known and if ``damp == 0``,
    one could proceed as follows:

      1. Compute a residual vector ``r0 = b - A*x0``.
      2. Use LSQR to solve the system  ``A*dx = r0``.
      3. Add the correction dx to obtain a final solution ``x = x0 + dx``.

    This requires that ``x0`` be available before and after the call
    to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
    to solve A*x = b and k2 iterations to solve A*dx = r0.
    If x0 is "good", norm(r0) will be smaller than norm(b).
    If the same stopping tolerances atol and btol are used for each
    system, k1 and k2 will be similar, but the final solution x0 + dx
    should be more accurate.  The only way to reduce the total work
    is to use a larger stopping tolerance for the second system.
    If some value btol is suitable for A*x = b, the larger value
    btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.

    Preconditioning is another way to reduce the number of iterations.
    If it is possible to solve a related system ``M*x = b``
    efficiently, where M approximates A in some helpful way (e.g. M -
    A has low rank or its elements are small relative to those of A),
    LSQR may converge more rapidly on the system ``A*M(inverse)*z =
    b``, after which x can be recovered by solving M*x = z.

    If A is symmetric, LSQR should not be used!

    Alternatives are the symmetric conjugate-gradient method (cg)
    and/or SYMMLQ.  SYMMLQ is an implementation of symmetric cg that
    applies to any symmetric A and will converge more rapidly than
    LSQR.  If A is positive definite, there are other implementations
    of symmetric cg that require slightly less work per iteration than
    SYMMLQ (but will take the same number of iterations).

    Original Copyright
    ------------------
    Sparse Equations and Least Squares.

    The original Fortran code was written by C. C. Paige and M. A. Saunders as
    described in

    C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
    equations and sparse least squares, TOMS 8(1), 43--71 (1982).

    C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
    equations and least-squares problems, TOMS 8(2), 195--209 (1982).

    It is licensed under the following BSD license:

    Copyright (c) 2006, Systems Optimization Laboratory
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

        * Redistributions of source code must retain the above copyright
          notice, this list of conditions and the following disclaimer.

        * Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials provided
          with the distribution.

        * Neither the name of Stanford University nor the names of its
          contributors may be used to endorse or promote products derived
          from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The Fortran code was translated to Python for use in CVXOPT by Jeffery
    Kline with contributions by Mridul Aanjaneya and Bob Myhill.

    Adapted for SciPy by Stefan van der Walt.

    References
    ----------
    .. [1] C. C. Paige and M. A. Saunders (1982a).
           "LSQR: An algorithm for sparse linear equations and
           sparse least squares", ACM TOMS 8(1), 43-71.
    .. [2] C. C. Paige and M. A. Saunders (1982b).
           "Algorithm 583.  LSQR: Sparse linear equations and least
           squares problems", ACM TOMS 8(2), 195-209.
    .. [3] M. A. Saunders (1995).  "Solution of sparse rectangular
           systems using LSQR and CRAIG", BIT 35, 588-604.

    """
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
         'Ax - b is small enough, given atol, btol                  ',
         'The least-squares solution is good enough, given atol     ',
         'The estimate of cond(Abar) has exceeded conlim            ',
         'Ax - b is small enough for this machine                   ',
         'The least-squares solution is good enough for this machine',
         'Cond(Abar) seems to be too large for this machine         ',
         'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
        str2 = 'damp = %20.14e   calc_var = %8g' % (damp, calc_var)
        str3 = 'atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
        str4 = 'btol = %8.2e               iter_lim = %8g' % (btol, iter_lim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Assign default vector norms, if necessary
    if unorm is None: unorm = norm
    if vnorm is None: vnorm = norm

    """
    Set up the first vectors u and v for the bidiagonalization.
    These satisfy  beta*u = b,  alfa*v = A'u.
    """
    v = np.zeros(n)
    u = b
    x = np.zeros(n)
    alfa = 0
    beta = unorm(u)
    w = np.zeros(n)

    if beta > 0:
        u = (1/beta) * u
        v = A.rmatvec(u)
        alfa = vnorm(v)

    if alfa > 0:
        v = (1/alfa) * v
        w = v.copy()

    rhobar = alfa
    phibar = beta
    bnorm = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(str1, str2, str3)

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        """
        %     Perform the next step of the bidiagonalization to obtain the
        %     next  beta, u, alfa, v.  These satisfy the relations
        %                beta*u  =  a*v   -  alfa*u,
        %                alfa*v  =  A'*u  -  beta*v.
        """
        u = A.matvec(v) - alfa * u
        beta = unorm(u)

        if beta > 0:
            u = (1/beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            v = A.rmatvec(u) - beta * v
            alfa = vnorm(v)
            if alfa > 0:
                v = (1 / alfa) * v

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + vnorm(dk)**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= iter_lim-10:
            prnt = True
        # if itn%10 == 0: prnt = True
        if test3 <= 2*ctol:
            prnt = True
        if test2 <= 10*atol:
            prnt = True
        if test1 <= 10*rtol:
            prnt = True
        if istop != 0:
            prnt = True

        if prnt:
            if show:
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (anorm, acond)
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
        str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
        str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
        str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, unorm=None, vnorm=None):
    """Iterative solver for least-squares problems.

    lsmr solves the system of linear equations ``Ax = b``. If the system
    is inconsistent, it solves the least-squares problem ``min ||b - Ax||_2``.
    A is a rectangular matrix of dimension m-by-n, where all cases are
    allowed: m = n, m > n, or m < n. B is a vector of length m.
    The matrix A may be dense or sparse (usually sparse).

    Parameters
    ----------
    A : {matrix, sparse matrix, ndarray, LinearOperator}
        Matrix A in the linear system.
    b : array_like, shape (m,)
        Vector b in the linear system.
    damp : float
        Damping factor for regularized least-squares. `lsmr` solves
        the regularized least-squares problem::

         min ||(b) - (  A   )x||
             ||(0)   (damp*I) ||_2

        where damp is a scalar.  If damp is None or 0, the system
        is solved without regularization.
    atol, btol : float, optional
        Stopping tolerances. `lsmr` continues iterations until a
        certain backward error estimate is smaller than some quantity
        depending on atol and btol.  Let ``r = b - Ax`` be the
        residual vector for the current approximate solution ``x``.
        If ``Ax = b`` seems to be consistent, ``lsmr`` terminates
        when ``norm(r) <= atol * norm(A) * norm(x) + btol * norm(b)``.
        Otherwise, lsmr terminates when ``norm(A^{T} r) <=
        atol * norm(A) * norm(r)``.  If both tolerances are 1.0e-6 (say),
        the final ``norm(r)`` should be accurate to about 6
        digits. (The final x will usually have fewer correct digits,
        depending on ``cond(A)`` and the size of LAMBDA.)  If `atol`
        or `btol` is None, a default value of 1.0e-6 will be used.
        Ideally, they should be estimates of the relative error in the
        entries of A and B respectively.  For example, if the entries
        of `A` have 7 correct digits, set atol = 1e-7. This prevents
        the algorithm from doing unnecessary work beyond the
        uncertainty of the input data.
    conlim : float, optional
        `lsmr` terminates if an estimate of ``cond(A)`` exceeds
        `conlim`.  For compatible systems ``Ax = b``, conlim could be
        as large as 1.0e+12 (say).  For least-squares problems,
        `conlim` should be less than 1.0e+8. If `conlim` is None, the
        default value is 1e+8.  Maximum precision can be obtained by
        setting ``atol = btol = conlim = 0``, but the number of
        iterations may then be excessive.
    maxiter : int, optional
        `lsmr` terminates if the number of iterations reaches
        `maxiter`.  The default is ``maxiter = min(m, n)``.  For
        ill-conditioned systems, a larger value of `maxiter` may be
        needed.
    show : bool, optional
        Print iterations logs if ``show=True``.
    unorm, vnorm : callable f(v), optional
        Functions to compute 2-norms in the range of A (unorm) or in the
        solution space (vnorm). Each function should take a single vector
        argument and return its 2-norm as a floating-point value. If either is
        not provided, numpy.linalg.norm will be used by default.

    Returns
    -------
    x : ndarray of float
        Least-square solution returned.
    istop : int
        istop gives the reason for stopping::

          istop   = 0 means x=0 is a solution.
                  = 1 means x is an approximate solution to A*x = B,
                      according to atol and btol.
                  = 2 means x approximately solves the least-squares problem
                      according to atol.
                  = 3 means COND(A) seems to be greater than CONLIM.
                  = 4 is the same as 1 with atol = btol = eps (machine
                      precision)
                  = 5 is the same as 2 with atol = eps.
                  = 6 is the same as 3 with CONLIM = 1/eps.
                  = 7 means ITN reached maxiter before the other stopping
                      conditions were satisfied.

    itn : int
        Number of iterations used.
    normr : float
        ``norm(b-Ax)``
    normar : float
        ``norm(A^T (b - Ax))``
    norma : float
        ``norm(A)``
    conda : float
        Condition number of A.
    normx : float
        ``norm(x)``

    Notes
    -----
    .. versionadded:: 0.11.0

    Original Copyright
    ------------------
    Copyright (C) 2010 David Fong and Michael Saunders

    07 Jun 2010: Documentation updated
    03 Jun 2010: First release version in Python

    David Chin-lung Fong            clfong@stanford.edu
    Institute for Computational and Mathematical Engineering
    Stanford University

    Michael Saunders                saunders@stanford.edu
    Systems Optimization Laboratory
    Dept of MS&E, Stanford University.

    References
    ----------
    .. [1] D. C.-L. Fong and M. A. Saunders,
           "LSMR: An iterative algorithm for sparse least-squares problems",
           SIAM J. Sci. Comput., vol. 33, pp. 2950-2971, 2011.
           http://arxiv.org/abs/1006.0758
    .. [2] LSMR Software, http://web.stanford.edu/group/SOL/software/lsmr/

    """

    A = aslinearoperator(A)
    b = atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    # Assign default vector norms, if necessary
    if unorm is None: unorm = norm
    if vnorm is None: vnorm = norm

    msg = ('The exact solution is  x = 0                              ',
         'Ax - b is small enough, given atol, btol                  ',
         'The least-squares solution is good enough, given atol     ',
         'The estimate of cond(Abar) has exceeded conlim            ',
         'Ax - b is small enough for this machine                   ',
         'The least-squares solution is good enough for this machine',
         'Cond(Abar) seems to be too large for this machine         ',
         'The iteration limit has been reached                      ')

    hdg1 = '   itn      x(1)       norm r    norm A''r'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m, n = A.shape

    # stores the num of singular values
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print('The matrix A has %8g rows  and %8g cols' % (m, n))
        print('damp = %20.14e\n' % (damp))
        print('atol = %8.2e                 conlim = %8.2e\n' % (atol, conlim))
        print('btol = %8.2e             maxiter = %8g\n' % (btol, maxiter))

    u = b
    beta = unorm(u)

    v = zeros(n)
    alpha = 0

    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alpha = vnorm(v)

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = zeros(n)
    x = zeros(n)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules.
    normb = beta
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (normr, normar)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(''.join([str1, str2, str3]))

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  a*v   -  alpha*u,
        #        alpha*v  =  A'*u  -  beta*v.

        u = A.matvec(v) - alpha * u
        beta = unorm(u)

        if beta > 0:
            u = (1 / beta) * u
            v = A.rmatvec(u) - beta * v
            alpha = vnorm(v)
            if alpha > 0:
                v = (1 / alpha) * v

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar = h - (thetabar * rho / (rhoold * rhobarold)) * hbar
        x = x + (zeta / (rho * rhobar)) * hbar
        h = v - (thetanew / rho) * h

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = vnorm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = infty
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.

        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (normr, normar)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (normA, condA)
                print(''.join([str1, str2, str3, str4]))

        if istop > 0:
            break

    # Print the stopping condition.

    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print('istop =%8g    normr =%8.1e' % (istop, normr))
        print('    normA =%8.1e    normAr =%8.1e' % (normA, normar))
        print('itn   =%8g    condA =%8.1e' % (itn, condA))
        print('    normx =%8.1e' % (normx))
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx
