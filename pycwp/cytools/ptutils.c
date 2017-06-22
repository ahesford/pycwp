/* Copyright (c) 2017 Andrew J. Hesford. All rights reserved.
 * Restrictions are listed in the LICENSE file distributed with this package. */

#include <math.h>
#include <float.h>

typedef struct {
	double x, y, z;
} point;

const double realeps = 5.1448789686149945e-12;

point axpy(double a, point x, point y) {
	/* Return a point struct equal to (a * x + y). */
	point r;
	r.x = a * x.x + y.x;
	r.y = a * x.y + y.y;
	r.z = a * x.z + y.z;
	return r;
}

point lintp(double t, point x, point y) {
	/* Return the linear interpolation (1 - t) * x + t * y. */
	point r;
	double m = 1 - t;
	r.x = m * x.x + t * y.x;
	r.y = m * x.y + t * y.y;
	r.z = m * x.z + t * y.z;
	return r;
}

point *iaxpy(double a, point x, point *y) {
	/* Store, in y, the value of (a * x + y). */
	y->x += a * x.x;
	y->y += a * x.y;
	y->z += a * x.z;
	return y;
}

point scal(double a, point x) {
	/* Scale and return the point x by a. */
	point r;
	r.x = x.x * a;
	r.y = x.y * a;
	r.z = x.z * a;
	return r;
}

point *iscal(double a, point *x) {
	/* Scale the point x, in place, by a. */
	x->x *= a;
	x->y *= a;
	x->z *= a;
	return x;
}

point *iptmpy(point h, point *x) {
	/* Compute, in x, the Hadamard product of points h and x. */
	x->x *= h.x;
	x->y *= h.y;
	x->z *= h.z;
	return x;
}

point *iptdiv(point h, point *x) {
	/* Compute, in x, the coordinate-wise ratio x / h. */
	x->x /= h.x;
	x->y /= h.y;
	x->z /= h.z;
	return x;
}

double ptnrm(point x) {
	/* The L2-norm of the point x. */
	double ns = x.x * x.x + x.y * x.y + x.z * x.z;
	return sqrt(ns);
}

double ptsqnrm(point x) {
	/* The squared L2-norm of the point x. */
	return x.x * x.x + x.y * x.y + x.z * x.z;
}

double ptdst(point x, point y) {
	/* The Euclidean distance between points x and y. */
	double dx, dy, dz;
	dx = x.x - y.x;
	dy = x.y - y.y;
	dz = x.z - y.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

double dot(point l, point r) {
	/* Return the inner product of two point structures. */
	return l.x * r.x + l.y * r.y + l.z * r.z;
}

point cross(point l, point r) {
	/* Return the cross product of two Point3D objects. */
	point o;
	o.x = l.y * r.z - l.z * r.y;
	o.y = l.z * r.x - l.x * r.z;
	o.z = l.x * r.y - l.y * r.x;
	return o;
}

int almosteq(double x, double y) {
	/* Returns True iff the difference between x and y is less than or
	 * equal to M * eps, where M = max(abs(x), abs(y), 1.0) and eps is the
	 * geometric mean of FLT_EPSILON and DBL_EPSILON. */
	double mxy = fmax(fabs(x), fmax(fabs(y), 1.0));
	return fabs(x - y) <= realeps * mxy;
}

double infdiv(double a, double b) {
	/* Return a / b with special handling of small values:
	 *
	 * 1. If |b| <= eps * |a|, return signed infinity,
	 * 2. Otherwise, if |a| <= eps, return 0,
	 *
	 * where eps is the geometric mean of FLT_EPSILON and DBL_EPSILON. */
	double aa = fabs(a), ab = fabs(b);

	if (ab <= realeps * aa) {
		if ((a >= 0) == (b >= 0)) return (1.0 / 0.0);
		else return -(1.0 / 0.0);
	} else if (aa <= realeps) return 0.0;

	return a / b;
}
