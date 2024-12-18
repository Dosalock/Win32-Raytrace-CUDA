#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

using BYTE = unsigned char;

class CUDAManager
{
public:
	// Singleton access
	static CUDAManager &GetInstance();

	// Prevent copying
	CUDAManager(const CUDAManager &) = delete;
	CUDAManager &operator=(const CUDAManager &) = delete;

	// Initialize CUDA device
	bool Initialize();

	// Get device information
	struct DeviceInfo
	{
		int deviceId;
		std::string name;
		size_t totalMemory;
		int computeCapability;
	};

	// Retrieve available device information
	std::vector<DeviceInfo> GetAvailableDevices();

	// Select best device based on capabilities
	int SelectBestDevice();

	// Perform core GPU computations
	
	void ProcessGPUData(BYTE *input, BYTE *output, size_t size);

	// Utility methods
	size_t GetDeviceTotalMemory() const;
	int GetDeviceComputeCapability() const;

private:
	// Private constructor for singleton
	CUDAManager() = default;

	// Error handling utility
	void CheckCUDAError(cudaError_t err, const char *context);

	// Device properties storage
	cudaDeviceProp m_device_prop{};
	int m_selected_device_id = -1;
};
