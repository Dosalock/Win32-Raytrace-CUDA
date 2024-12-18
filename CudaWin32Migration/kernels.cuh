#pragma once
#define _USE_MATH_DEFINES

#include <Windows.h>
// #include <cuda_runtime.h>

#include "object_structs.h"
#include <cuda.h>
#include <tuple>

void WrapperKernel ( BYTE *input, BYTE *output, dim3 a, dim3 b );
//__global__ void cuda_Draw ( BYTE *lpv_bits_in, BYTE *lpv_bits_out );
__global__ void cuda_Draw ( BYTE *pLpvBits,
							Camera *cam,
							Sphere *device_scene,
							Light *device_lights );

__host__ void Draw_Caller ( BYTE **pLpvBits, Camera *cam );

__device__ Intersection ClosestIntersection ( float4 O,
											  float4 D,
											  float t_min,
											  float t_max,
											  Sphere *device_scene );
