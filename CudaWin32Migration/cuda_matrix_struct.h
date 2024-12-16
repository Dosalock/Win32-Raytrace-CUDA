#pragma once

#include <cuda_fp16.h>

struct float3x3
{
	float4 m00_m01_m02;
	float4 m10_m11_m12;
	float4 m20_m21_m22;

	__host__ __device__ float3x3(float4 m00_m01_m02,
								 float4 m10_m11_m12,
							     float4 m20_m21_m22)
	{
		this->m00_m01_m02 = m00_m01_m02;
		this->m10_m11_m12 = m10_m11_m12;
		this->m20_m21_m22 = m20_m21_m22;
	}
};
