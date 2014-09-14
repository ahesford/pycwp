#define FLOAT2(vec) { (float) (vec).x, (float) (vec).y }

/* Use linear interpolation to resample the source image src into dst, where
 * the destination grid is rotated an angle theta (in radians) about the center
 * of the source image grid. The destination grid spacing is given by (dx,dy)
 * and the source grid spacing is given by (sx,sy). */
__kernel void rotinterp(__write_only image2d_t dst,
				     const float dx, const float dy,
				     __read_only image2d_t src,
				     const float sx, const float sy,
				     const float theta) {
	/* Grab the index into the work item. */
	const int2 idx = { get_global_id(0), get_global_id(1) };

	/* Grab image dimensions and ensure destination pixel exists. */
	const int2 srcshape = get_image_dim(src);
	const int2 dstshape = get_image_dim(dst);

	if (idx.x < 0 || idx.x >= dstshape.x || 
		  idx.y < 0 || idx.y >= dstshape.y) return;

	/* Build vectors of grid spacings. */
	const float2 ddlt = { dx, dy };
	const float2 sdlt = { sx, sy };

	/* Convert coordinates and dimensions to floats. */
	const float2 fidx = FLOAT2(idx);
	const float2 fsrcshape = FLOAT2(srcshape);
	const float2 fdstshape = FLOAT2(dstshape);

	const float2 one = (float2) (1.0f);
	const float2 two = (float2) (2.0f);

	/* Find the pixel origins for the source and destination. */
	const float2 dstorig = (fdstshape - one) / two;

	/* Convert pixel coordinates into physical coordinates. */
	const float2 crd = ddlt * (fidx - dstorig);

	/* Rotate the coordinates as desired. */
	float cosa;
	const float sina = sincos(theta, &cosa);
	const float2 rcrd = { mad(cosa, crd.x, -sina * crd.y),
				mad(sina, crd.x, cosa * crd.y) };

	/* Convert back to unnormalized pixel coordinates. The origin shift
	 * should be (fsrcshape - 1) / two, but OpenCL defines read_imagef 
	 * such that the extra 1/2 is subtracted automatically. */
	const float2 scrd = rcrd / sdlt + fsrcshape / two;

	/* Read the interpolated value and write to destination. */
	const sampler_t lin = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;
	const float4 pval = read_imagef(src, lin, scrd);
	write_imagef(dst, idx, pval);
}
