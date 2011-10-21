<%!
	import math
%>

float2 integrate(const float, const float3, const float3);

/* 3-D integration of the far-field pattern for a source and direction. */
float2 integrate(const float k, const float3 src, const float3 obs) {
	float3 spt;
	float2 val, ans = (float2) (0.);
	uint i, j, l;

	float pts[${len(pts)}], wts[${len(wts)}];

	## Populate the arrays of points and weights
	% for i, (p, w) in enumerate(zip(pts, wts)):
		pts[${i}] = ${p};
		wts[${i}] = ${w};
	% endfor

	/* Compute the contributions to the intergration. */
	for (i = 0; i < ${len(pts)}; ++i) {
		spt.x = src.x + 0.5 * ${dc} * pts[i];
		for (j = 0; j < ${len(pts)}; ++j) {
			spt.y = src.y + 0.5 * ${dc} * pts[j];
			for (l = 0; l < ${len(pts)}; ++l) {
				spt.z = src.z + 0.5 * ${dc} * pts[l];
				val.x = cos(k * dot(spt, obs));
				val.y = sin(k * dot(spt, obs));
				ans += (float2) (wts[i] * wts[j] * wts[l]) * val;
			}
		}
	}
	ans *= (float2) (${dc**3 / 8.});

	return ans;
}

__kernel void farmat(__global float2 * const mat, __global float * const theta,
		__global float * const phi, const float k, const float3 src) {
	/* Get the global grid position. */
	const uint gi = get_global_id(0);
	const uint gj = get_global_id(1);

	/* Get the global grid size. */
	const uint np = get_global_size(0);
	const uint nt = get_global_size(1) + 2;

	/* Find the work item's angular position. */
	const float thval = theta[gj + 1];
	const float phval = phi[gi];

	const float st = sin(thval);

	/* Find the direction corresponding to the work item. */
	const float3 obs = (float3) (cos(phval) * st, sin(phval) * st, cos(thval));

	float2 ans;

	/* Integrate the source and observation. */
	ans = integrate(k, src, obs);
	/* Offset the value by one to account for the pole. */
	mat[gj * np + gi + 1] = ans;

	/* The first work item computes the polar values. */
	if (gi == 0 && gj == 0) {
		/* The first pole comes first. */
		mat[0] = integrate(k, src, (float3) (0., 0., cos(theta[0])));

		/* The last pole comes last. */
		mat[(nt - 2) * np + 1] = 
			integrate(k, src, (float3) (0., 0., cos(theta[nt - 1])));
	}
}
