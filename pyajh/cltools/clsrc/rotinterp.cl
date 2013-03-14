#define FLOAT2(vec) { (float) (vec).x, (float) (vec).y }

/* Use linear interpolation to resample the source image src into dst, where
 * the destination grid is rotated an angle theta (in radians) about the center
 * of the source image grid. The source image is contracted by factors cx in x
 * and cy in y before the resampling and rotation operations. */
__kernel void rotinterp(__write_only image2d_t dst, __read_only image2d_t src,
		const float theta, const float cx, const float cy) {
	/* Grab the index into the work item. */
	const int2 idx = { get_global_id(0), get_global_id(1) };

	/* Grab image dimensions and ensure destination pixel exists. */
	const int2 srcshape = get_image_dim(src);
	const int2 dstshape = get_image_dim(dst);

	if (idx.x < 0 || idx.x >= dstshape.x || 
		  idx.y < 0 || idx.y >= dstshape.y) return;

	const float2 h = (float2) (0.5f);

	const float2 fidx = FLOAT2(idx);
	const float2 fsrcshape = FLOAT2(srcshape);
	const float2 fdstshape = FLOAT2(dstshape);

	/* Convert pixel coordinates into normalized range [-0.5, 0.5]. */
	const float2 crd = (fidx + h) / fdstshape - h;

	/* Rotate the coordinates into the source axes. */
	float cosa;
	const float sina = sincos(theta, &cosa);
	/* Include the x and y contraction factors. */
	const float2 rcrd = { cx * mad(cosa, crd.x, -sina * crd.y),
				cy * mad(sina, crd.x, cosa * crd.y) };

	/* Convert back to unnormalized pixel coordinates.
	 * Although a half-pixel should be subtracted, the OpenCL
	 * spec folds this into the definition of read_imagef. */
	const float2 scrd = (rcrd + h) * fsrcshape;

	/* Read the interpolated value and write to destination. */
	const sampler_t lin = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;
	const float4 pval = read_imagef(src, lin, scrd);
	write_imagef(dst, idx, pval);
}
