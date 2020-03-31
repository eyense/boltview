// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/mgt/resource_constraints.h>


#include <array>
#include <future>
#include <vector>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

class GpuMemoryManager;
// TODO(johny) - handle cuda streams in non-cuda code differently
#if !defined(__CUDACC__)
using cudaStream_t = void*;
#endif  // __CUDACC__


/// A Task is a part of the Problem that fits on a single gpu,
/// the task should be run asynchronously on a given stream.
class ITask {
public:
	using GroupId = uint32_t;

	virtual ~ITask() = default;

	/// Allocate all buffers for the task
	virtual void allocate(GpuMemoryManager* gpu_memory_manager) = 0;

	/// Starts asynchronously the task on the given stream - necessary
	/// allocations by gpu memory manager.
	virtual void execute(cudaStream_t stream) = 0;

	/// Tasks with the same task_group_id should ideally run on the same gpu stream
	/// presumably beacause of shared input or output. Scheduler will assign tasks to
	/// different gpu workers based on this value.
	virtual GroupId getGroupId() const = 0;
};

/// A Problem defines some work that needs to be done on one or more gpus. It splits
/// into multiple tasks, each of which can be run on a single gpu, granularity of this
/// splitting can be adjusted.
class IProblem {
public:
	using ProblemId = uint32_t;

	virtual ~IProblem() = default;
	/// split the problem into tasks that can fit on any given gpu by themselves
	virtual std::vector<std::unique_ptr<ITask>> generateTasks(ResourceConstraints) = 0;

	virtual ProblemId getProblemId() const = 0;
};

/// @}

}  // namespace mgt

}  // namespace bolt
