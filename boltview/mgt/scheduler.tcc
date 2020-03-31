// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se
#pragma once

#include <boltview/mgt/gpu_worker.h>

#include <future>
#include <vector>

namespace bolt {

namespace mgt {


namespace detail {

inline std::vector<int> selectGpuIds(SelectGpu select_gpu) {
	auto devices = device::getCudaCapableDevices();
	if(select_gpu == SelectGpu::FIRST_GPU) {
		return std::vector<int>(1, devices[0]);
	} else {
		return devices;
	}
}

}  // namespace detail


inline SchedulerConfiguration::SchedulerConfiguration(
		std::vector<int> gpu_ids,
		ThreadsPerGpu num_threads_per_gpu,
		UseGpuMemoryFraction use_gpu_memory_fraction)
{
	for (int gpu_id : gpu_ids) {
		worker_configurations_.push_back({gpu_id, num_threads_per_gpu, use_gpu_memory_fraction});
	}
}

inline SchedulerConfiguration::SchedulerConfiguration(
		SelectGpu select_gpu,
		ThreadsPerGpu num_threads_per_gpu,
		UseGpuMemoryFraction use_gpu_memory_fraction) :
			SchedulerConfiguration(detail::selectGpuIds(select_gpu),
			num_threads_per_gpu,
			use_gpu_memory_fraction)
{}

inline std::vector<DeviceProperties> SchedulerConfiguration::getAvailableResourcesForThread() const {
	std::vector<DeviceProperties> device_properties;
	for (const GpuWorkerConfiguration& configuration : worker_configurations_) {
		DeviceProperties config_device_properties = configuration.getDeviceProperties();
		config_device_properties.total_memory /= configuration.getThreadsPerGpu();
		device_properties.push_back(config_device_properties);
	}
	return device_properties;
}


inline SchedulerConfiguration getSchedulerConfiguration(int rank, int num_ranks, ThreadsPerGpu threads_per_gpu, UseGpuMemoryFraction use_gpu_fraction, int max_available_gpus) {
	std::vector<int> gpu_ids;
	std::vector<int> all_gpu_ids = device::getCudaCapableDevices();
	int num_gpus = std::min((int)all_gpu_ids.size(), max_available_gpus);
	if (num_gpus > num_ranks) {
		for (int i = rank; i < num_gpus; i += num_ranks) {
			gpu_ids.push_back(all_gpu_ids[i]);
		}
	} else if (num_ranks % num_gpus == 0) {
		gpu_ids.push_back(all_gpu_ids[rank % num_gpus]);
		use_gpu_fraction.get() /= (num_ranks / num_gpus);
	} else {
		BOLT_THROW(MultithreadLaunchError() << bolt::MessageErrorInfo("Number of ranks must be divisible by number of gpus"));
	}
	return SchedulerConfiguration(gpu_ids, threads_per_gpu, use_gpu_fraction);
}

/// A helper struct to hold data related to a single Problem and manage its lifetime.
/// This struct is held by the Scheduler until all the Tasks generated from the given Problem
/// are finished, then the wrapper is destructed.
struct ActiveProblem {
	std::unique_ptr<IProblem> problem;
	std::vector<ITask::GroupId> task_group_ids;
	std::vector<std::future<TaskStatus>> task_futures;
	std::promise<void> promise;
};


struct Scheduler::PImpl {
	void runWorkers();
	void runAsync();
	void stopAsync();
	std::future<void> addProblem(std::unique_ptr<IProblem> problem);

	void waitForProblemsToFinish();
	void waitForProblemsToFinishAsync(int thread_id);
	void waitForSingleProblemToFinish(ActiveProblem&& problem);

	std::vector<GpuWorkerConfiguration> worker_configurations_;

	std::vector<GpuWorker> gpu_workers_;
	ResourceConstraints resource_constraints_;

	std::vector<std::thread> waiting_threads_;
	std::queue<ActiveProblem> problem_queue_;

	bool shutdown_;
	std::mutex problem_queue_mutex_;
	std::condition_variable problem_added_;

};

inline void Scheduler::PImpl::runAsync() {
	if (waiting_threads_.size() == 0) {
		shutdown_ = false;
		waiting_threads_.push_back(std::thread([this]() { waitForProblemsToFinishAsync(0); }));
	}
	runWorkers();
}

inline void Scheduler::PImpl::stopAsync() {
	if (waiting_threads_.size() > 0) {
		{
			std::lock_guard<std::mutex> lock(problem_queue_mutex_);
			shutdown_ = true;
		}
		problem_added_.notify_all();
		for (auto& thread : waiting_threads_) {
			thread.join();
		}
	}
}


inline void Scheduler::PImpl::runWorkers() {
	for (unsigned int i = 0; i < worker_configurations_.size(); i++) {
		gpu_workers_[i].runAsync(worker_configurations_[i]);
	}
}

inline void Scheduler::PImpl::waitForProblemsToFinish() {
	while(problem_queue_.size() > 0) {
		ActiveProblem active = std::move(problem_queue_.front());
		problem_queue_.pop();
		BOLT_DFORMAT("Waiting for problem with problem_id = %1%", active.problem->getProblemId());
		auto promise = std::move(active.promise);
		waitForSingleProblemToFinish(std::move(active));
		promise.set_value();
	}
}

inline void Scheduler::PImpl::waitForProblemsToFinishAsync(int thread_id) {
	while(true) {
		ActiveProblem active;
		{
			std::unique_lock<std::mutex> lock(problem_queue_mutex_);
			problem_added_.wait(lock, [&]{ return shutdown_ || problem_queue_.size() > 0; });
			if (shutdown_ && problem_queue_.size() == 0) {
				return;
			}
			active = std::move(problem_queue_.front());
			problem_queue_.pop();
		}

		BOLT_DFORMAT("Waiting for problem with problem_id = %1% on thread = %2%", active.problem->getProblemId(), thread_id);
		auto promise = std::move(active.promise);
		try {
			waitForSingleProblemToFinish(std::move(active));
			promise.set_value();
		} catch (const BoltError& e) {
			promise.set_exception(std::current_exception());
		}
	}
}

inline void Scheduler::PImpl::waitForSingleProblemToFinish(ActiveProblem&& active) {
	for (unsigned int i = 0; i < active.task_futures.size(); i++){
		BOLT_DFORMAT("Waiting for task with group_id = %1%", active.task_group_ids[i]);
		active.task_futures[i].get();
	}
}

inline std::future<void> Scheduler::PImpl::addProblem(std::unique_ptr<IProblem> problem) {
	ActiveProblem active;
	auto tasks = problem->generateTasks(resource_constraints_);
	for (auto& task : tasks) {
		int task_group_id = task->getGroupId();
		active.task_group_ids.push_back(task_group_id);

		int target_worker = task_group_id % gpu_workers_.size();
		active.task_futures.push_back(
			gpu_workers_[target_worker].addTask(std::move(task)));
	}

	active.problem = std::move(problem);
	std::future<void> problem_future = active.promise.get_future();

	{
		std::lock_guard<std::mutex> lock(problem_queue_mutex_);
		problem_queue_.push(std::move(active));
	}
	problem_added_.notify_all();
	return problem_future;
}


inline Scheduler::Scheduler(SchedulerConfiguration config) :
	pimpl_(new PImpl{
			config.worker_configurations_,
			std::vector<GpuWorker>(config.getNumberOfGpus()),
			ResourceConstraints(config.getAvailableResourcesForThread()) })
{}

inline std::future<void> Scheduler::addProblem(std::unique_ptr<IProblem> problem) {
	return pimpl_->addProblem(std::move(problem));
}

inline Scheduler::~Scheduler() {
	stopAsync();
}

inline void Scheduler::runAsync() {
	pimpl_->runAsync();
}

inline void Scheduler::stopAsync() {
	pimpl_->stopAsync();
}

inline int Scheduler::getNumberOfProblemsInQueue() {
	std::lock_guard<std::mutex> lock(pimpl_->problem_queue_mutex_);
	return 	pimpl_->problem_queue_.size();
}


inline void Scheduler::runUntilFinished() {
	pimpl_->runWorkers();
	pimpl_->waitForProblemsToFinish();
}


}  // namespace mgt

}  // namespace bolt
