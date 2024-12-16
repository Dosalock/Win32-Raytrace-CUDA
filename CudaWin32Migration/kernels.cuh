#pragma once
#define _USE_MATH_DEFINES

#include <Windows.h>
//#include <cuda_runtime.h>


#include "object_structs.h"
#include "cuda_vector_functions.h"



__host__ void Draw_Caller(BYTE **pLpvBits, Camera *cam);

__device__ Intersection ClosestIntersection(float4 O,
											float4 D,
											float t_min,
											float t_max,
											Sphere *device_scene);
