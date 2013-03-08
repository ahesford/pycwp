<%
	dim = len(srcshape)

	if dim == 2:
		image = 'image2d_t'
		idxtype = 'int2'
		crdtype = 'float2'
	elif dim == 3:
		image = 'image3d_t'
		idxtype = 'int3'
		crdtype = 'float3'
%>

## Print the tuple as a floating-point number
<%def name="prtuple(p)">
	${'(' + ', '.join(str(float(pv)) + 'f' for pv in p) + ')'}
</%def>

__kernel void lininterp(__write_only ${image} dst, __read_only ${image} src) {
	const sampler_t lin = CLK_FILTER_LINEAR | CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP;
	/* Grab the index into the work item. */
	const ${idxtype} idx = (${idxtype}) (get_global_id(0), get_global_id(1)
		${', get_global_id(2)' if dim == 3 else ''} );
	const ${crdtype} fidx = (${crdtype}) ((float) idx.x, (float) idx.y
		${', (float) idx.z' if dim == 3 else ''} );

	const ${crdtype} half = (${crdtype}) (0.5f);
	const ${crdtype} dstshape = (${crdtype}) ${prtuple(dstshape)};
	const ${crdtype} srcshape = (${crdtype}) ${prtuple(srcshape)};

	/* Compute the scale between destination and source pixels. */
	const ${crdtype} scale = srcshape / dstshape;
	/* comvert destination coordinates to source. */
	const ${crdtype} crd = scale * (fidx + half) - half;

	/* Read the interpolated value and write to the destination. */
	const float4 pval = read_imagef(src, lin, crd);
	write_imagef(dst, idx, pval);
}
