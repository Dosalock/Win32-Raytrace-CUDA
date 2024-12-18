#pragma once


#include <Windows.h>
#include <cmath>
#include <cuda_fp16.h>

struct Camera
{
	float4 position;
	float yaw;
	float pitch;
	float roll;
};

struct Sphere
{
	float4 center;
	float radius;
	COLORREF color;
	int specularity;
	float reflective;
	float radius_squared;

	__device__ Sphere ( float4 center    = { },
			 float radius     = 0,
			 COLORREF color   = RGB( 0, 0, 0 ),
			 int specularity  = 0,
			 float reflective = 0 )
	{
		this->center         = center;
		this->radius         = radius;
		this->color          = color;
		this->specularity    = specularity;
		this->reflective     = reflective;
		this->radius_squared = radius * radius;
	}
};

struct Light
{
	enum LightType
	{
		directional,
		point,
		ambient
	} type;

	float intensity;
	float4 pos;
};

struct Intersection
{
	Sphere *sphere;
	float point;

	__device__ Intersection ( Sphere *sphere = NULL, float point = INFINITY )
	{
		this->sphere = sphere;
		this->point  = point;
	}
};
