#include "kernels.cuh"


#include "cuda_vector_functions.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>

#define HEIGHT     1024
#define WIDTH      1024
#define NUM_LIGHTS 3

void CreateScene ( Sphere *device_scene, Light *device_lights )
{
	device_scene[0].center = make_float4( 0, -1, 3 );
	device_scene[0].radius = 1;
	device_scene[0].color =
	  RGB( 255, 0, 0 ); // Replace with device compatible type or logic
	device_scene[0].specularity = 500;
	device_scene[0].reflective  = 0.2;
	device_scene[0].radius_squared =
	  device_scene[0].radius * device_scene[0].radius;

	device_scene[1].center = make_float4( 2, 0, 4 );
	device_scene[1].radius = 1;
	device_scene[1].color =
	  RGB( 0, 0, 255 ); // Replace with device compatible type or logic
	device_scene[1].specularity = 500;
	device_scene[1].reflective  = 0.3;
	device_scene[1].radius_squared =
	  device_scene[1].radius * device_scene[1].radius;

	device_scene[2].center = make_float4( -2, 0, 4 );
	device_scene[2].radius = 1;
	device_scene[2].color =
	  RGB( 0, 255, 0 ); // Replace with device compatible type or logic
	device_scene[2].specularity = 10;
	device_scene[2].reflective  = 0.4;
	device_scene[2].radius_squared =
	  device_scene[2].radius * device_scene[2].radius;

	device_scene[3].center = make_float4( 0, -5001, 0 );
	device_scene[3].radius = 5000;
	device_scene[3].color =
	  RGB( 255, 255, 0 ); // Replace with device compatible type or logic
	device_scene[3].specularity = 1000;
	device_scene[3].reflective  = 0.5;
	device_scene[3].radius_squared =
	  device_scene[3].radius * device_scene[3].radius;


	device_lights[0].type      = device_lights->ambient;
	device_lights[0].intensity = 0.2;

	device_lights[1].type      = device_lights->point;
	device_lights[1].intensity = 0.6;
	device_lights[1].pos       = make_float4( 2, 1, 0 );

	device_lights[2].type      = device_lights->directional;
	device_lights[2].intensity = 0.2;
	device_lights[2].pos       = make_float4( 1, 4, 4 );
}

__device__ bool IntersectionBounds ( float T, float t_min, float t_max )
{
	return ( T > t_min && T < t_max ); // Strict inequality
}

__device__ float4 ReflectRay ( float4 R, float4 N )
{
	return ( 2.0f * dot( N, R ) * N ) - R;
}

__device__ float CalcLight ( float4 P,
							 float4 N,
							 float4 V,
							 int s,
							 Sphere *device_scene,
							 Light *device_lights )
{
	float intensity = 0.0;
	float t_max     = 0;
	float4 L        = { };
	float4 R        = { };
	for ( int i = 0; i < NUM_LIGHTS; i++ )
	{
		if ( device_lights[i].type == device_lights->ambient )
		{
			intensity += device_lights[i].intensity;
		}
		else
		{
			if ( device_lights[i].type == device_lights->point )
			{
				L     = ( device_lights[i].pos - P );
				t_max = 1;
			}
			else
			{
				L     = device_lights[i].pos;
				t_max = INFINITY;
			}
			L           = L;
			float t_min = 0.0001f;
			Intersection res =
			  ClosestIntersection( P, L, t_min, t_max, device_scene );
			Sphere *closest_sphere = res.sphere;
			if ( closest_sphere != NULL )
			{
				continue;
			}

			float n_dot_l = dot( N, L );
			if ( n_dot_l > 0 )
			{
				intensity += device_lights[i].intensity * n_dot_l
							 / ( len( N ) * len( L ) );
			}

			if ( s != -1 )
			{
				R             = ReflectRay( L, N );
				float r_dot_v = dot( R, V );

				if ( r_dot_v > 0 )
				{
					intensity +=
					  device_lights[i].intensity
					  * pow( r_dot_v / ( len( R ) * ( len( V ) ) ), s );
				}
			}
		}
	}
	return intensity;
}

__device__ float
  IntersectRaySphere ( float4 O, float4 D, Sphere sphere, float dDot )
{
	float4 CO = { };
	CO        = O - sphere.center;

	float a = dot(D,D);
	float b = 2 * dot( CO, D );
	float c = dot( CO, CO ) - sphere.radius_squared;

	float discr = b * b - 4 * a * c;

	if ( discr < 0 )
	{
		return INFINITY;
	}
	else if ( discr == 0 )
	{
		return -b / ( 2 * a );
	}

	float t = ( -b - sqrtf( discr ) )
			  / ( 2 * a ); // Minimize compute only go for 1 root;

	return t;
}

__device__ Intersection ClosestIntersection ( float4 O,
											  float4 D,
											  float t_min,
											  float t_max,
											  Sphere *device_scene )
{
	float closest_t        = INFINITY;
	Sphere *closest_sphere = NULL;
	float d_dot_d          = dot( D, D ); // Cache immutable value


	for ( int i = 0; i < 4; i++ )
	{
		float t = IntersectRaySphere( O, D, device_scene[i], d_dot_d );

		if ( IntersectionBounds( t, t_min, t_max ) && t < closest_t )
		{
			closest_t      = t;
			closest_sphere = const_cast<Sphere *>( &device_scene[i] );
		}
	}
	return Intersection( closest_sphere, closest_t );
}

__device__ COLORREF TraceRay ( float4 O,
							   float4 D,
							   float t_min,
							   float t_max,
							   int recursionDepth,
							   Sphere *device_scene,
							   Light *device_lights )
{
	float4 N = { };
	float4 P = { };
	float4 R = { };

	Intersection intersec =
	  ClosestIntersection( O, D, t_min, t_max, device_scene );
	Sphere *closest_sphere = intersec.sphere;
	float closest_t        = intersec.point;
	if ( closest_sphere == NULL )
	{
		return RGB( 0, 0, 0 );
	}

	P = O + ( closest_t * D );
	N = normalize( P - closest_sphere->center );

	float res = CalcLight( P,
						   N,
						   invert( D ),
						   closest_sphere->specularity,
						   device_scene,
						   device_lights );
	int r     = ( int )round( GetRValue( closest_sphere->color ) * res );
	int g     = ( int )round( GetGValue( closest_sphere->color ) * res );
	int b     = ( int )round( GetBValue( closest_sphere->color ) * res );

	float refl = closest_sphere->reflective;

	if ( recursionDepth <= 0 || refl <= 0 )
	{
		return RGB( max( 0, min( 255, r ) ),
					max( 0, min( 255, g ) ),
					max( 0, min( 255, b ) ) );
	}

	R                       = ReflectRay( invert( D ), N );
	COLORREF reflectedColor = TraceRay( P,
										R,
										t_min,
										t_max,
										recursionDepth - 1,
										device_scene,
										device_lights );

	int reflected_r = ( int )roundf( GetRValue( reflectedColor ) ) * refl;
	int reflected_g = ( int )roundf( GetGValue( reflectedColor ) ) * refl;
	int reflected_b = ( int )roundf( GetBValue( reflectedColor ) ) * refl;


	return RGB(
	  max( 0, min( 255, static_cast<int>( r * ( 1 - refl ) + reflected_r ) ) ),
	  max( 0, min( 255, static_cast<int>( g * ( 1 - refl ) + reflected_g ) ) ),
	  max( 0,
		   min( 255, static_cast<int>( b * ( 1 - refl ) + reflected_b ) ) ) );
}

__device__ float4 CanvasToViewPort ( int x, int y )
{
	// for simplicity : Vw = Vh = d = 1    approx 53 fov
	float aspectRatio = static_cast<float>( WIDTH ) / HEIGHT;

	// x and y to the viewport, adjusting by aspect ratio
	float fovMod = 1;
	float viewportX =
	  ( x - WIDTH / 2.0 ) * ( ( 1.0 * fovMod ) / WIDTH ) * aspectRatio;
	float viewportY =
	  -( y - HEIGHT / 2.0 )
	  * ( ( 1.0 * fovMod ) / HEIGHT ); // Flip Y to match 3D space orientation

	return make_float4( viewportX,
						viewportY,
						1,
						0 ); // Z=1 for perspective projection
}

//__global__ void cuda_Draw(BYTE *lpv_bits_in, BYTE *lpv_bits_out)
//{
//	int x = blockIdx.x * blockDim.x + threadIdx.x;
//	int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//	int x_scaled = static_cast<int>( ( x * 255.0f ) / 1024.0f );
//	int y_scaled = static_cast<int>( ( y * 255.0f ) / 1024.0f );
//
//	int offset = ( y * WIDTH + x ) * 4;
//	if ( offset >= 0 && offset < WIDTH * HEIGHT * 4 - 4 )
//	{
//		lpv_bits_out[offset + 0] = ( int )GetBValue( 255 * x_scaled  );
//		lpv_bits_out[offset + 1] = ( int )GetGValue( 255 * y_scaled  );
//		lpv_bits_out[offset + 2] = ( int )GetRValue( 255 * x_scaled );
//		lpv_bits_out[offset + 3] = 255;
//	}
//}

__global__ void cuda_Draw ( BYTE *pLpvBits,
							Camera *cam,
							Sphere *device_scene,
							Light *device_lights )
{
	float4 D           = { };
	float t_min        = 0.0001;
	float t_max        = INFINITY;
	int recursionDepth = 0;
	Camera local_cam   = *cam;



	int pixel_debug_offset = 0;
#ifdef _SINGLE_THREAD_DEBUG
	pixel_debug_offset = 700;
#endif

	int x = blockIdx.x * blockDim.x + threadIdx.x + pixel_debug_offset;
	int y = blockIdx.y * blockDim.y + threadIdx.y + pixel_debug_offset;

	printf("X coordinate: %d \n Y coordinate: %d \n", x, y);
	
	D = normalize( CanvasToViewPort( x, y ) );
	/*D = normalize( apply_camera_rotation( local_cam.position,
										  local_cam.yaw,
										  local_cam.pitch,
										  local_cam.roll ) );*/

	COLORREF color = TraceRay( local_cam.position,
							   D,
							   t_min,
							   t_max,
							   recursionDepth,
							   device_scene,
							   device_lights );

	D = normalize( D );

	int offset = ( y * WIDTH + x ) * 4;
	if ( offset >= 0 && offset < WIDTH * HEIGHT * 4 - 4 )
	{
		pLpvBits[offset + 0] = ( int )GetBValue( color );
		pLpvBits[offset + 1] = ( int )GetGValue( color );
		pLpvBits[offset + 2] = ( int )GetRValue( color );
		pLpvBits[offset + 3] = 255;
	}
}

void MakeWorld ( Sphere *&scene_ptr, Light *&lights_ptr )
{
	Sphere *device_scene;
	Light *device_lights;
	cudaMalloc( ( void ** )&device_scene, sizeof( Sphere ) * 4 );
	cudaMalloc( ( void ** )&device_lights, sizeof( Light ) * NUM_LIGHTS );

	Sphere *host_scene = ( Sphere * )malloc( sizeof( Sphere ) * 4 );
	Light *host_lights = ( Light * )malloc( sizeof( Light ) * NUM_LIGHTS );


	CreateScene( host_scene, host_lights );

	cudaMemcpy( device_scene,
				host_scene,
				sizeof( Sphere ) * 4,
				cudaMemcpyHostToDevice );
	cudaMemcpy( device_lights,
				host_lights,
				sizeof( Light ) * NUM_LIGHTS,
				cudaMemcpyHostToDevice );
	scene_ptr  = device_scene;
	lights_ptr = device_lights;
	free( host_scene );
	free( host_lights );
}

void WrapperKernel ( BYTE *input, BYTE *output, dim3 a, dim3 b )
{
	Sphere *scene = nullptr;
	Light *lights = nullptr;
	MakeWorld( scene, lights );

	Camera cam = { };
	Camera *cuda_cam;
	cudaMalloc( ( void ** )&cuda_cam, sizeof( Camera ) );
	cudaMemcpy( cuda_cam, &cam, sizeof( Camera ), cudaMemcpyHostToDevice );

	cuda_Draw<<<a, b>>>( output, cuda_cam, scene, lights );

	cudaDeviceSynchronize( );
	cudaFree( scene );
	cudaFree( lights );
}

__host__ void Draw_Caller ( BYTE **pLpvBits, Camera *cam )
{
	int N = 1024;
	dim3 threadsPB( 16, 16 );
	dim3 numB( N / threadsPB.x, N / threadsPB.y );


	BYTE *cudaLpvBits;
	size_t src_pitch = ( ( WIDTH * 4 + 3 ) & ~3 );
	size_t dest_pitch;

	cudaMallocPitch( &cudaLpvBits,
					 &dest_pitch,
					 WIDTH * 4 * sizeof( BYTE ),
					 HEIGHT );
	cudaMemcpy2D( cudaLpvBits,
				  dest_pitch,
				  *pLpvBits,
				  src_pitch,
				  WIDTH * 4 * sizeof( BYTE ),
				  HEIGHT,
				  cudaMemcpyHostToDevice );


	Camera *cuda_cam;
	cudaMalloc( ( void ** )&cuda_cam, sizeof( Camera ) );
	cudaMemcpy( cuda_cam, cam, sizeof( Camera ), cudaMemcpyHostToDevice );


	Sphere *device_scene;
	Light *device_lights;
	cudaMalloc( ( void ** )&device_scene, sizeof( Sphere ) * 4 );
	cudaMalloc( ( void ** )&device_lights, sizeof( Light ) * NUM_LIGHTS );

	Sphere *host_scene = ( Sphere * )malloc( sizeof( Sphere ) * 4 );
	Light *host_lights = ( Light * )malloc( sizeof( Light ) * NUM_LIGHTS );


	CreateScene( host_scene, host_lights );

	cudaMemcpy( device_scene,
				host_scene,
				sizeof( Sphere ) * 4,
				cudaMemcpyHostToDevice );
	cudaMemcpy( device_lights,
				host_lights,
				sizeof( Light ) * NUM_LIGHTS,
				cudaMemcpyHostToDevice );

	// cuda_Draw << <numB, threadsPB >> > (cudaLpvBits, cuda_cam, device_scene,
	// device_lights);


	cudaDeviceSynchronize( );
	cudaMemcpy( cam,
				( void * )cuda_cam,
				sizeof( Camera ),
				cudaMemcpyDeviceToHost );

	cudaMemcpy2D( *pLpvBits,
				  src_pitch,
				  cudaLpvBits,
				  dest_pitch,
				  WIDTH * 4 * sizeof( BYTE ),
				  HEIGHT,
				  cudaMemcpyDeviceToHost );

	cudaFree( cudaLpvBits );
	cudaFree( cuda_cam );
	cudaFree( device_scene );
	cudaFree( device_lights );
}