// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#include <chrono>
#include <vector>

#include <boltview/copy.h>
#include <boltview/math/vector.h>
#include <boltview/procedural_views.h>

#include "scheduler_test_problems.h"


namespace bolt {

namespace mgt {

void SleepTask::execute(cudaStream_t stream) {
	std::this_thread::sleep_for(std::chrono::milliseconds(size_));
}

std::unique_ptr<ITask> makeSleepTask(int size) {
	return std::unique_ptr<ITask>(new SleepTask(0, size));
}

void AllocationTask::allocate(GpuMemoryManager* gpu) {
	image  = gpu->getBuffer(TextureImageDescriptor<float, 2>(Int2(100, 100)));
	image2 = gpu->getBuffer(DeviceImageDescriptor<float, 2>(Int2(100, 100)));
	image3 = gpu->getBuffer(DeviceImageDescriptor<int, 2>(Int2(100, 100)));
}

void InfiniteAllocationTask::allocate(GpuMemoryManager* gpu) {
	std::vector<GpuBufferView<ImageType>> buffers;
	while(true) {
		auto size = ImageType::SizeType::fill(512);
		buffers.emplace_back(gpu->getBuffer(ImageDescriptor<ImageType>(size)));
	}
}

void OversizedTextureTask::allocate(GpuMemoryManager* gpu) {
	Int2 size(65536 * 8, 2);
	auto buffer = gpu->getBuffer(ImageDescriptor<TextureImageType>(size));
}

void DoubleArrayOnCpuTask::execute(cudaStream_t stream) {
	for (int i = 0; i < array_view_.size(); i++) {
		array_view_[i] *= 2;
	}
}

void FftTask::allocate(GpuMemoryManager* gpu) {
	Int3 size(128, 128, 20);
	Int3 fft_size = getFftImageSize(size);
	forward_calculator = gpu->getBuffer(FftCalculatorDescriptor<1, DeviceFftPolicy<Forward, Stack<1,2>>>(size));
	inverse_calculator = gpu->getBuffer(FftCalculatorDescriptor<1, DeviceFftPolicy<Inverse, Stack<1,2>>>(size));
	image_real = gpu->getBuffer(DeviceImageDescriptor<float, 3>(size));
	image_complex = gpu->getBuffer(DeviceImageDescriptor<cufftComplex, 3>(fft_size));
}

void FftTask::execute(cudaStream_t stream) {
	forward_calculator->setStream(stream);
	inverse_calculator->setStream(stream);
	forward_calculator->calculate(image_real->view(), image_complex->view());
	inverse_calculator->calculate(image_complex->view(), image_real->view());
}

std::vector<std::unique_ptr<ITask>> DoubleArrayOnCpuProblem::generateTasks(ResourceConstraints) {
	std::vector<std::unique_ptr<ITask>> tasks;
	int size_of_task = array_view_.size() / num_tasks_;
	for (int i = 0; i < array_view_.size(); i += size_of_task) {
		auto subview = array_view_.subview(i, size_of_task);
		std::unique_ptr<ITask> task_ptr(new DoubleArrayOnCpuTask(i / size_of_task, subview));
		tasks.push_back(std::move(task_ptr));
	}
	return tasks;
}

void DoubleImageOnGpuTask::allocate(GpuMemoryManager* gpu) {
	slice_gpu_ = gpu->getBuffer(ImageDescriptor<DeviceImage<int, 2>>(slice_view_.size()));
}

void DoubleImageOnGpuTask::execute(cudaStream_t stream) {
	auto slice_gpu_view_ = slice_gpu_->view();
	auto doubled_view = multiplyByFactor(2, slice_gpu_view_);

	copyAsync(slice_view_, slice_gpu_view_, stream);
	copyAsync(doubled_view, slice_gpu_view_, stream);
	copyAsync(slice_gpu_view_, slice_view_, stream);
	BOLT_CHECK(cudaStreamSynchronize(stream));
}

std::vector<std::unique_ptr<ITask>> DoubleImageOnGpuProblem::generateTasks(ResourceConstraints) {
	std::vector<std::unique_ptr<ITask>> tasks;
	int num_tasks = image_view_.size()[2];
	for (int i = 0; i < num_tasks; i++) {
		auto slice = image_view_.slice<2>(i);
		std::unique_ptr<ITask> task_ptr(new DoubleImageOnGpuTask(i, slice));
		tasks.push_back(std::move(task_ptr));
	}
	return tasks;
}


void SharedBufferTask::execute(cudaStream_t stream) {
	HostImage<float, 2> image(image_size_);
	HostImageView<float, 2> image_view = image.view();
	DeviceImageView<float, 2> image_gpu_view = image_gpu_->view();

	if(image_gpu_.isFresh() != should_be_fresh_){
		BOLT_THROW(AssertionFailedSharedBufferTaskFreshExpectation());
	}
	if (image_gpu_.isFresh()) {
		// if image is fresh (new memory was allocated), store value_ to [0, 0] element
		image_view[Int2(0, 0)] = value_;
		copyAsync(image_view, image_gpu_view, stream);
	} else {
		// else check, if value_ is already in [0, 0] element
		copyAsync(image_gpu_view, image_view, stream);
		if(image_view[Int2(0, 0)] != value_){
			BOLT_THROW(AssertionFailedSharedBufferTaskInitialElementExpectation());
		}
	}
	BOLT_CHECK(cudaStreamSynchronize(stream));
}


void SharedBufferTask::allocate(GpuMemoryManager* gpu) {
	image_gpu_ = gpu->getSharedBuffer(DeviceImageDescriptor<float, 2>(image_size_), id_);
}


void NotSharedBufferTask::execute(cudaStream_t stream) {
	HostImage<float, 2> image(image_size_);
	HostImageView<float, 2> image_view = image.view();
	DeviceImageView<float, 2> image_gpu_view = image_gpu_->view();
	image_view[Int2(0, 0)] = value_;
	copyAsync(image_view, image_gpu_view, stream);
	BOLT_CHECK(cudaStreamSynchronize(stream));
}


void NotSharedBufferTask::allocate(GpuMemoryManager* gpu) {
	image_gpu_ = gpu->getBuffer(DeviceImageDescriptor<float, 2>(image_size_));
}


std::vector<std::unique_ptr<ITask>> SharedBufferProblem::generateTasks(ResourceConstraints) {
	const Int2 image_size(10, 10);
	const uint64_t id1 = 1, id2 = 2;
	const float shared_value1 = 10.0f, shared_value2 = 20.0f;
	const float value = 30.0f;
	bool should_be_fresh;

	std::vector<std::unique_ptr<ITask>> tasks;

	// task no. 0
	should_be_fresh = true;
	tasks.emplace_back(new SharedBufferTask(0, id1, image_size, shared_value1, should_be_fresh));

	// task no. 1
	// shared_cache is not overriden by a NotSharedBufferTask with the different id
	tasks.emplace_back(new NotSharedBufferTask(0, image_size, value));

	// task no. 2
	// shared_cache is not overriden by a SharedBufferTask with the different id
	should_be_fresh = true;
	tasks.emplace_back(new SharedBufferTask(0, id2, image_size, shared_value2, should_be_fresh));

	// task no. 3
	// right shared_cache is used
	should_be_fresh = false;
	tasks.emplace_back(new SharedBufferTask(0, id1, image_size, shared_value1, should_be_fresh));

	return tasks;
}


}  // namespace mgt

}  // namespace bolt
