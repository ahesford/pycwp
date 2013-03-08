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

__kernel void farmat(__global float2 * const mat, const float k,
			const float3 src, __global float2 * const angles) {
	/* Get the global position. */
	const uint idx = get_global_id(0);
	/* Grab the angular coordinates for the work item. */
	const float2 curang = angles[idx];

	/* The first entry is the polar angle, the second the azimuthal angle. */
	const float thval = curang.s0, phval = curang.s1;

	/* Find the direction corresponding to the work item. */
	const float st = sin(thval);
	const float3 obs = (float3) (cos(phval) * st, sin(phval) * st, cos(thval));

	/* Integrate the source and observation. */
	mat[idx] = integrate(k, src, obs);
}
