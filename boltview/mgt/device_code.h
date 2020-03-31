// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#if !defined(__CUDACC__)
	#error device_code.h cannot be included in file which is not compiled by nvcc!
#endif

#include <cuda_runtime.h>
#include <boltview/exceptions.h>
#include <boltview/fft/fft_calculator.h>
#include <boltview/cuda_utils.h>
#include <boltview/mgt/device_code.h>
#include <vector>

namespace bolt {

namespace mgt {
namespace device {

/// \addtogroup Utilities
/// @{

/// separate all cuda calls to a single translation unit


inline std::vector<int> getCudaCapableDevices() {
	return ::bolt::getCudaCapableDevices();
}

inline DeviceMemoryInfo getDeviceMemoryInfo() {
	return ::bolt::getDeviceMemoryInfo();
}

inline DeviceProperties getDeviceProperties(int gpu_id) {
	return ::bolt::getDeviceProperties(gpu_id);
}

inline void setCudaDevice(int gpu_id) {
	BOLT_CHECK(cudaSetDevice(gpu_id));
}

/// RAII wrapper for cudaStream_t
class CudaStream {
public:
	CudaStream() {
		BOLT_CHECK(cudaStreamCreate(&stream_));
	}

	~CudaStream() {
		BOLT_CHECK(cudaStreamDestroy(stream_));
	}

	CudaStream(const CudaStream&) = delete;
	CudaStream& operator= (const CudaStream&) = delete;

	cudaStream_t get() {
		return stream_;
	}

private:
	cudaStream_t stream_;
};

/// @}

/// \addtogroup mgt Multi GPU Scheduling
/// @{

struct VoidPointerDeleter {
	void operator()(void* ptr) {
		BOLT_CHECK(cudaFree(ptr));
	}

};

struct CudaArrayDeleter {
	void operator()(cudaArray* array){
		BOLT_CHECK(cudaFreeArray(array));
	}

};

struct CufftPlanDeleter {
	template<typename TType>
	void operator()(TType* plan) {
		BOLT_CUFFT_CHECK(cufftDestroy(*plan));
	}

};

/// @}

}  // namespace device
}  // namespace mgt

}  // namespace bolt
