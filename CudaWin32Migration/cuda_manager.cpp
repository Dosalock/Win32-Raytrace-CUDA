#include "cuda_manager.h"
#include "kernels.cuh"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>

#ifdef _WIN32
#include <Windows.h>
#endif

// Singleton instance getter
CUDAManager &CUDAManager::GetInstance ( )
{
	static CUDAManager instance;
	return instance;
}

// Comprehensive error checking method
void CUDAManager::CheckCUDAError ( cudaError_t err, const char *context )
{
	if ( err != cudaSuccess )
	{
		std::stringstream error_msg;
		error_msg << "CUDA Error in " << context << ": "
				  << cudaGetErrorString( err ) << " (Error Code: " << err
				  << ")";

		// Debug output
#ifdef _DEBUG
		OutputDebugStringA( error_msg.str( ).c_str( ) );
#endif
		time_t current_time = time( nullptr );
		struct tm local_time;
		char buffer[80];
		// File logging
		std::ofstream log_file( "cuda_error.log", std::ios_base::app );
		if ( log_file.is_open( ) )
		{
			localtime_s( &local_time, &current_time );
			strftime( buffer,
					  sizeof( buffer ),
					  "%Y-%m-%d %H:%M:%S",
					  &local_time );
			log_file << "[ " << buffer << " ]: " << error_msg.str( )
					 << std::endl;
			log_file.close( );
		}

		// Throw exception for critical errors
		throw std::runtime_error( error_msg.str( ) );
	}
}

// Device initialization
bool CUDAManager::Initialize ( )
{
	// Check for CUDA devices

	int device_count = 0;
	cudaError_t err  = cudaGetDeviceCount( &device_count );

	CheckCUDAError( err, "cudaGetDeviceCount" );

	if ( device_count == 0 )
	{
		throw std::runtime_error( "No CUDA-capable devices found" );
	}

	// Select and set the best device
	m_selected_device_id = SelectBestDevice( );
	err                  = cudaSetDevice( m_selected_device_id );
	CheckCUDAError( err, "cudaSetDevice" );

	// Retrieve device properties
	err = cudaGetDeviceProperties( &m_device_prop, m_selected_device_id );
	CheckCUDAError( err, "cudaGetDeviceProperties" );

	return true;
}

// Select the best CUDA device
int CUDAManager::SelectBestDevice ( )
{
	int best_device         = 0;
	int max_multiprocessors = 0;

	int device_count = 0;
	cudaGetDeviceCount( &device_count );

	for ( int device = 0; device < device_count; ++device )
	{
		cudaDeviceProp properties;
		cudaGetDeviceProperties( &properties, device );

		// Selection criteria: most multiprocessors
		if ( properties.multiProcessorCount > max_multiprocessors )
		{
			max_multiprocessors = properties.multiProcessorCount;
			best_device         = device;
		}
	}

	return best_device;
}

// Get available CUDA devices
std::vector<CUDAManager::DeviceInfo> CUDAManager::GetAvailableDevices ( )
{
	std::vector<DeviceInfo> devices;
	int device_count = 0;
	cudaGetDeviceCount( &device_count );

	for ( int i = 0; i < device_count; ++i )
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties( &prop, i );

		DeviceInfo info;
		info.deviceId          = i;
		info.name              = prop.name;
		info.totalMemory       = prop.totalGlobalMem;
		info.computeCapability = prop.major * 10 + prop.minor;

		devices.push_back( info );
	}

	return devices;
}

// Process data on GPU

void CUDAManager::ProcessGPUData ( BYTE *input, BYTE *output, size_t size )
{
	BYTE *d_input  = nullptr;
	BYTE *d_output = nullptr;

	try
	{
		// Allocate device memory
		cudaError_t err = cudaMalloc( &d_input, size );
		CheckCUDAError( err, "cudaMalloc input" );

		err = cudaMalloc( &d_output, size );
		CheckCUDAError( err, "cudaMalloc output" );

		// Copy input to device
		err = cudaMemcpy( d_input, input, size, cudaMemcpyHostToDevice );
		CheckCUDAError( err, "cudaMemcpy H2D" );


		// Launch kernel

		// TODO: get this to work for non square viewports

		// Amount of bytes to pixel height
		int screen_pixel_dimension = static_cast<int>( sqrtf( size ) ) >> 1;

		dim3 threads_per_block( 16, 16 );
		dim3 blocks_per_grid(64,64);

		size_t src_pitch = ( ( screen_pixel_dimension * 4 + 3 ) & ~3 );
		size_t dest_pitch;

		// Allocate device memory
		err = cudaMallocPitch( &d_output,
							   &dest_pitch,
							   size / screen_pixel_dimension,
							   screen_pixel_dimension );
		CheckCUDAError( err, "cudaMallocPitch 2device" );

		// Copy input to device
		err = cudaMemcpy2D( d_input,
							dest_pitch,
							input,
							src_pitch,
							size / screen_pixel_dimension,
							screen_pixel_dimension,
							cudaMemcpyHostToDevice );
		CheckCUDAError( err, "cudaMemcpy2D" );


		WrapperKernel( d_input, d_output, blocks_per_grid, threads_per_block );

		// err = cudaDeviceSynchronize( );
		CheckCUDAError( err, "cudaDeviceSynchronize" );

		err = cudaMemcpy2D( output,
							src_pitch,
							d_output,
							dest_pitch,
							size / screen_pixel_dimension,
							screen_pixel_dimension,
							cudaMemcpyDeviceToHost );
		CheckCUDAError( err, "cudaMemcpy2D 2host" );
	}
	catch ( const std::exception &e )
	{
		// Clean up resources in case of error
		if ( d_input )
		{
			cudaFree( d_input );
		}
		if ( d_output )
		{
			cudaFree( d_output );
		}
		throw;
	}

	// Free device memory
	if ( d_input )
	{
		cudaFree( d_input );
	}
	if ( d_output )
	{
		cudaFree( d_output );
	}
}

// Utility methods
size_t CUDAManager::GetDeviceTotalMemory ( ) const
{
	return m_device_prop.totalGlobalMem;
}

int CUDAManager::GetDeviceComputeCapability ( ) const
{
	return m_device_prop.major * 10 + m_device_prop.minor;
}