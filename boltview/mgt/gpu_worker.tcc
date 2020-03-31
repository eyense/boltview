// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se
#pragma once

#include <boltview/mgt/device_code.h>
#include <boltview/mgt/task_interface.h>

#include <future>
#include <vector>

namespace bolt {

namespace mgt {


inline GpuWorkerConfiguration::GpuWorkerConfiguration(int gpu_id, ThreadsPerGpu num_threads_per_gpu, UseGpuMemoryFraction use_gpu_memory_fraction) :
	gpu_id_(gpu_id),
	num_threads_per_gpu_(num_threads_per_gpu.get())
{
	device_properties_ = device::getDeviceProperties(gpu_id);
	device_properties_.total_memory *= use_gpu_memory_fraction.get();
}


inline std::vector<std::future<TaskStatus>> GpuWorker::addTasks(std::vector<std::unique_ptr<ITask>> tasks) {
	std::vector<std::future<TaskStatus>> futures;
	{
		std::lock_guard<std::mutex> lock(task_queue_mutex_);
		for (auto& task : tasks) {
			ActiveTask active {std::move(task), {}};
			futures.push_back(active.promise.get_future());
			task_queue_.push(std::move(active));
		}
	}
	task_added_.notify_all();
	return futures;
}

inline std::future<TaskStatus> GpuWorker::addTask(std::unique_ptr<ITask> task) {
	std::future<TaskStatus> future;
	{
		std::lock_guard<std::mutex> lock(task_queue_mutex_);
		ActiveTask active {std::move(task), {}};
		future = active.promise.get_future();
		task_queue_.push(std::move(active));
	}
	task_added_.notify_all();
	return future;
}

inline void GpuWorker::runAsync(GpuWorkerConfiguration configuration) {
	if (worker_threads_.size() == 0) {
		gpu_memory_manager_.setGpu(
			configuration.gpu_id_,
			configuration.device_properties_.total_memory);
		for (int i = 0; i < configuration.num_threads_per_gpu_; i++) {
			worker_threads_.push_back(std::thread([=]() { worker(i, configuration.gpu_id_); }));
		}
	}
}

inline GpuWorker::~GpuWorker() {
	{
		std::lock_guard<std::mutex> lock(task_queue_mutex_);
		shutdown_ = true;
	}
	task_added_.notify_all();
	for (auto& thread : worker_threads_) {
		thread.join();
	}
}

inline void GpuWorker::worker(int thread_id, int gpu_id) {
	device::setCudaDevice(gpu_id);
	device::CudaStream stream;
	while(true) {
		ActiveTask active;
		{
			std::unique_lock<std::mutex> lock(task_queue_mutex_);
			task_added_.wait(lock, [&]{ return shutdown_ || task_queue_.size() > 0; });
			if (shutdown_ && task_queue_.size() == 0) {
				return;
			}
			active = std::move(task_queue_.front());
			task_queue_.pop();
		}
		BOLT_DFORMAT("Running task with group_id = %1% on gpu %2% and thread %3%", active.task->getGroupId(), gpu_id, thread_id);
		try {
			active.task->allocate(&gpu_memory_manager_);  // DEBUG_BREAK
			active.task->execute(stream.get());
			active.promise.set_value(TaskStatus::COMPLETED);
		} catch (const BoltError& e) {
			active.promise.set_exception(std::current_exception());
		}
	}
}

}  // namespace mgt

}  // namespace bolt
