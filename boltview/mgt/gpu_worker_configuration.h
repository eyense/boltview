// Copyright 2018 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <boltview/numeric_wrapper.h>
#include <boltview/device_properties.h>
#include <boltview/exceptions.h>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

struct MgtError: BoltError {};

using ThreadsPerGpu = NumericTypeWrapper<int, struct threads_per_gpu>;
using UseGpuMemoryFraction = NumericTypeWrapper<double, struct gpu_mem_fraction>;

class GpuWorkerConfiguration {
public:
	GpuWorkerConfiguration(int gpu_id, ThreadsPerGpu num_threads_per_gpu, UseGpuMemoryFraction use_gpu_memory_fraction);

	DeviceProperties getDeviceProperties() const {
		return device_properties_;
	}

	int getThreadsPerGpu() const {
		return num_threads_per_gpu_;
	}

	friend class GpuWorker;

private:
	int gpu_id_;
	int num_threads_per_gpu_;
	DeviceProperties device_properties_;
};

/// @}

}  // namespace mgt

}  // namespace bolt
