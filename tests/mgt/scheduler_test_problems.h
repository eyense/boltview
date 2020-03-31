// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/array_view.h>
#include <boltview/fft/fft_utils.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/mgt/task_base.h>
#include <boltview/mgt/gpu_memory_manager.h>
#include <boltview/texture_image.h>

#include <vector>

namespace bolt {

namespace mgt {

/// contains problems and tasks used exclusively in scheduler_test

class SleepTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu_memory_manager) override {}
	void execute(cudaStream_t stream) override;

	explicit SleepTask(GroupId task_group_id = 0, uint64_t size = 100ULL) :
		BaseTask(task_group_id, 0),
		size_(size)
	{}

private:
	uint64_t size_;
};

std::unique_ptr<ITask> makeSleepTask(int size);

class AllocationTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override {}

	explicit AllocationTask(GroupId task_group_id = 0) : BaseTask(task_group_id, 0) {}

private:
	GpuBufferView<TextureImage<float, 2>> image;
	GpuBufferView<DeviceImage<float, 2>> image2;
	GpuBufferView<DeviceImage<int, 2>> image3;
};

class InfiniteAllocationTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override {}

	explicit InfiniteAllocationTask(GroupId task_group_id = 0) : BaseTask(task_group_id, 0) {}

private:
	typedef DeviceImage<float, 3> ImageType;
};

class OversizedTextureTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override {}

	explicit OversizedTextureTask(GroupId task_group_id = 0) : BaseTask(task_group_id, 0) {}

private:
	typedef TextureImage<float, 2> TextureImageType;
};

class FftTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override;

	explicit FftTask(GroupId task_group_id = 0) : BaseTask(task_group_id, 0) {}

private:
	GpuBufferView<FftCalculator<1, DeviceFftPolicy<Forward, Stack<1,2>>>> forward_calculator;
	GpuBufferView<FftCalculator<1, DeviceFftPolicy<Inverse, Stack<1,2>>>> inverse_calculator;
	GpuBufferView<DeviceImage<float, 3>> image_real;
	GpuBufferView<DeviceImage<cufftComplex, 3>> image_complex;
};

template<typename TTask>
class TestProblem : public BaseProblem {
public:
	std::vector<std::unique_ptr<ITask>> generateTasks(ResourceConstraints) override {
		std::vector<std::unique_ptr<ITask>> tasks;
		for (int i = 0; i < num_tasks_; i++) {
			std::unique_ptr<ITask> task_ptr(new TTask(i));
			tasks.push_back(std::move(task_ptr));
		}
		return tasks;
	}

	explicit TestProblem(int num_tasks, ProblemId problem_id = 0) :
		BaseProblem(problem_id),
		num_tasks_(num_tasks)
	{}

private:
	int num_tasks_;
};

template<typename TTask>
std::unique_ptr<IProblem> makeTestProblem(int num_tasks) {
	return std::unique_ptr<IProblem>(new TestProblem<TTask>(num_tasks));
}

template<typename TTask>
std::vector<std::unique_ptr<IProblem>> makeListOfProblems(int num_problems) {
	std::vector<std::unique_ptr<IProblem>> problem_list;
	for (int i = 0; i < num_problems; i++) {
		problem_list.push_back(makeTestProblem<TTask>(10));
	}
	return problem_list;
}

class DoubleArrayOnCpuTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override {}
	void execute(cudaStream_t stream) override;

	explicit DoubleArrayOnCpuTask(GroupId task_group_id, HostArrayView<int> data_view) :
		BaseTask(task_group_id, 0),
		array_view_(data_view)
	{}

private:
	HostArrayView<int> array_view_;
};

class DoubleArrayOnCpuProblem : public BaseProblem {
public:
	std::vector<std::unique_ptr<ITask>> generateTasks(ResourceConstraints) override;

	explicit DoubleArrayOnCpuProblem(HostArrayView<int> data_view, int num_tasks, ProblemId problem_id = 0) :
		BaseProblem(problem_id),
		array_view_(data_view),
		num_tasks_(num_tasks)
	{}

private:
	int num_tasks_;
	HostArrayView<int> array_view_;
};

class DoubleImageOnGpuTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override;

	explicit DoubleImageOnGpuTask(GroupId task_group_id, HostImageView<int, 2> data_view) :
		BaseTask(task_group_id, 0),
		slice_view_(data_view)
	{}

private:
	HostImageView<int, 2> slice_view_;
	GpuBufferView<DeviceImage<int, 2>> slice_gpu_;
};

class DoubleImageOnGpuProblem : public BaseProblem {
public:
	std::vector<std::unique_ptr<ITask>> generateTasks(ResourceConstraints) override;

	explicit DoubleImageOnGpuProblem(HostImageView<int, 3> data_view, ProblemId problem_id = 0) :
		BaseProblem(problem_id),
		image_view_(data_view)
	{}

private:
	HostImageView<int, 3> image_view_;
};

struct AssertionFailedSharedBufferTaskFreshExpectation: BoltError {};
struct AssertionFailedSharedBufferTaskInitialElementExpectation: BoltError {};

class SharedBufferTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override;

	explicit SharedBufferTask(
		GroupId task_group_id,
		uint64_t id,
		const Int2 &image_size,
		float value,
		bool should_be_fresh
	) :
		BaseTask(task_group_id, 0), image_size_(image_size), id_(id), value_(value), should_be_fresh_(should_be_fresh)
	{}

private:
	GpuBufferView<DeviceImage<float, 2>> image_gpu_;
	uint64_t id_;
	const Int2 image_size_;
	float value_;
	bool should_be_fresh_;
};


class NotSharedBufferTask : public BaseTask {
public:
	void allocate(GpuMemoryManager* gpu) override;
	void execute(cudaStream_t stream) override;

	explicit NotSharedBufferTask(GroupId task_group_id, const Int2 &image_size, float value) :
		BaseTask(task_group_id, 0), image_size_(image_size), value_(value) {}

private:
	GpuBufferView<DeviceImage<float, 2>> image_gpu_;
	const Int2 image_size_;
	float value_;
};


class SharedBufferProblem : public BaseProblem {
public:
	std::vector<std::unique_ptr<ITask>> generateTasks(ResourceConstraints) override;

	explicit SharedBufferProblem(ProblemId problem_id) :
		BaseProblem(problem_id)
	{}
};


}  // namespace mgt

}  // namespace bolt
