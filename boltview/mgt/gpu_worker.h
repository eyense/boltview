// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/mgt/task_interface.h>
#include <boltview/mgt/gpu_memory_manager.h>
#include <boltview/mgt/gpu_worker_configuration.h>

#include <future>
#include <queue>
#include <thread>
#include <vector>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

/// This serves as a means to notify the scheduler of task completion
enum class TaskStatus { COMPLETED };

struct ActiveTask {
	std::unique_ptr<ITask> task;
	std::promise<TaskStatus> promise;
};

/// This class handles all tasks for a single GPU.
/// Each GpuWorker manages its own task queue, acquires memory for the tasks from
/// memory manager and executes them on some stream - multiple tasks can be executed concurrently.
/// For each stream a separate thread is launched that issues cuda commands.
class GpuWorker {
public:
	GpuWorker() : shutdown_(false) {}
	~GpuWorker();

	/// Enqueue tasks.
	/// \threadsafe
	std::vector<std::future<TaskStatus>> addTasks(std::vector<std::unique_ptr<ITask>> tasks);

	/// Enqueue task.
	/// \threadsafe
	std::future<TaskStatus> addTask(std::unique_ptr<ITask> task);

	/// Run a thread  for each stream that will execute tasks from the queue, non-blocking.
	void runAsync(GpuWorkerConfiguration configuration);

private:
	void worker(int thread_id, int gpu_id);

	GpuMemoryManager gpu_memory_manager_;
	std::vector<std::thread> worker_threads_;
	std::queue<ActiveTask> task_queue_;

	bool shutdown_;
	std::mutex task_queue_mutex_;
	std::condition_variable task_added_;

	std::array<char, 64> padding_;  // prevent false sharing between threads
};

/// @}

}  // namespace mgt

}  // namespace bolt

#include <boltview/mgt/gpu_worker.tcc>
