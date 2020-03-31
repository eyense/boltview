// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/mgt/scheduler_configuration.h>
#include <boltview/mgt/task_interface.h>

#include <future>
#include <queue>
#include <vector>
#include <memory>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{


/// A Scheduler manages a queue of Problems and schedules the Tasks generated from
/// them on multiple GPUs via GpuWorkers. It assumes ownership of gpu memory, so
/// more than one instance shouldn't be used at the same time in a single process.
///
/// There are two modes of operating the scheduler:
/// 1) scheduler.runAsync();
///    std::future f = scheduler.addProblem(problem);
///    f.get();
///      - this async mode can be safely used from multiple threads
/// 2) scheduler.addProblem(problem1);
///    scheduler.addProblem(problem2);
///    scheduler.runUntilFinished(); //(blocking call)
///      - single thread use only, once RunUntilFinished is started, it is the users
///        responsibility to ensure no further Problems are added from other threads
///      - note that the order of execution of problem1 and problem2 is not fixed
///
/// The scheduling behavior is very simple at the moment, assigning tasks evenly
/// across all gpu workers in the order in which they are accepted.
/// TODO(tom): assign tasks to gpu workers based on their current workload
class Scheduler {
public:

	explicit Scheduler(SchedulerConfiguration config = getDefaultSchedulerConfiguration());
	~Scheduler();

	/// Add problem and schedule it's execution. Problems may be
	/// executed in a different order than they are submitted.
	/// \threadsafe
	std::future<void> addProblem(std::unique_ptr<IProblem> problem);

	/// Start scheduler and its gpu workers threads, non-blocking
	/// \threadsafe
	void runAsync();

	/// Start gpu workers threads + block until all queued problems are finished
	/// \NOTthreadsafe
	void runUntilFinished();

	int getNumberOfProblemsInQueue();

	/// Call this if you want to switch to single-thread use after using the async api
	void stopAsync();

private:
	struct PImpl;
	std::unique_ptr<PImpl> pimpl_;
};

/// @}

}  // namespace mgt

}  // namespace bolt

#include <boltview/mgt/scheduler.tcc>
