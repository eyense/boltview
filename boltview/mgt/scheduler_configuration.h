// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <algorithm>

#include <boltview/mgt/gpu_worker_configuration.h>

#include <vector>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

enum class SelectGpu { FIRST_GPU, ALL_GPUS };

struct MultithreadLaunchError : MgtError {};


class SchedulerConfiguration {
public:
	explicit SchedulerConfiguration(
		std::vector<int> gpu_ids,
		ThreadsPerGpu num_threads_per_gpu = ThreadsPerGpu(2),
		UseGpuMemoryFraction use_gpu_memory_fraction = UseGpuMemoryFraction(0.75));

	explicit SchedulerConfiguration(
		SelectGpu select_gpu,
		ThreadsPerGpu num_threads_per_gpu = ThreadsPerGpu(2),
		UseGpuMemoryFraction use_gpu_memory_fraction = UseGpuMemoryFraction(0.75));

	int getNumberOfGpus() const {
		return worker_configurations_.size();
	}

	std::vector<DeviceProperties> getAvailableResourcesForThread() const;

	friend class Scheduler;

private:
	std::vector<GpuWorkerConfiguration> worker_configurations_;
};

inline SchedulerConfiguration getDefaultSchedulerConfiguration() {
	return SchedulerConfiguration(SelectGpu::ALL_GPUS, ThreadsPerGpu(2), UseGpuMemoryFraction(0.75));
}

SchedulerConfiguration getSchedulerConfiguration(int rank, int num_ranks, ThreadsPerGpu threads_per_gpu, UseGpuMemoryFraction use_gpu_fraction, int max_available_gpus);

/// @}

}  // namespace mgt

}  // namespace bolt
