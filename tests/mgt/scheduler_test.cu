// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se
#define BOOST_TEST_MODULE SchedulerTest
#include <boost/test/unit_test.hpp>
#include <boost/test/included/unit_test.hpp>

#include <future>
#include <numeric>
#include <vector>


#include <boltview/array_view.h>
#include <boltview/host_image.h>
#include <boltview/math/vector.h>
#include <boltview/mgt/gpu_worker.h>
#include <boltview/mgt/scheduler.h>
#include "scheduler_test_problems.h"

#include "../test_utils.h"

namespace bolt {

namespace mgt {

void testSchedulerOnSingleProblemAsync(SchedulerConfiguration scheduler_configuration, std::unique_ptr<IProblem> dummy_problem) {
	Scheduler scheduler(scheduler_configuration);
	scheduler.runAsync();
	std::future<void> status = scheduler.addProblem(std::move(dummy_problem));
	BOOST_CHECK(status.valid());
	status.get();
	BOOST_CHECK(scheduler.getNumberOfProblemsInQueue() == 0);
}

void testSchedulerOnSingleProblem(SchedulerConfiguration scheduler_configuration, std::unique_ptr<IProblem> dummy_problem) {
	Scheduler scheduler(scheduler_configuration);
	scheduler.addProblem(std::move(dummy_problem));
	scheduler.runUntilFinished();
	BOOST_CHECK(scheduler.getNumberOfProblemsInQueue() == 0);
}

void testWorkerOnSingleTaskAsync(std::unique_ptr<ITask> dummy_task) {
	GpuWorker worker;
	worker.runAsync({0, ThreadsPerGpu(2), UseGpuMemoryFraction(1.0)});
	std::future<TaskStatus> status = worker.addTask(std::move(dummy_task));
	BOOST_CHECK(status.valid());
	BOOST_CHECK(status.get() == TaskStatus::COMPLETED);
}

void TestSchedulerOnMultipleProblems(
	SchedulerConfiguration scheduler_configuration,
	std::vector<std::unique_ptr<IProblem>> problems,
	bool sync_after_each_problem)
{
	Scheduler scheduler(scheduler_configuration);
	for (std::unique_ptr<IProblem>& problem : problems) {
		scheduler.addProblem(std::move(problem));
		if(sync_after_each_problem) {
			scheduler.runUntilFinished();
			BOOST_CHECK(scheduler.getNumberOfProblemsInQueue() == 0);
		}
	}
	if(!sync_after_each_problem) {
		scheduler.runUntilFinished();
		BOOST_CHECK(scheduler.getNumberOfProblemsInQueue() == 0);
	}
}

void TestDoubleArrayOnCpu(SchedulerConfiguration scheduler_configuration) {
	int num_gpus = scheduler_configuration.getNumberOfGpus();
	std::vector<int> array(num_gpus * 1000);
	std::iota(array.begin(), array.end(), 0);

	auto array_view = makeHostArrayView(array);

	std::unique_ptr<IProblem> problem(new DoubleArrayOnCpuProblem(array_view, num_gpus * 10));
	testSchedulerOnSingleProblemAsync(scheduler_configuration, std::move(problem));

	for (int i = 0; i < array_view.size(); i++) {
		BOOST_CHECK(array_view[i] == i * 2);
	}
}

void testDoubleImageOnGpu(SchedulerConfiguration scheduler_configuration) {
	int num_gpus = scheduler_configuration.getNumberOfGpus();
	HostImage<int, 3> image(Int3(100, 100, num_gpus * 10));
	HostImageView<int, 3> image_view = image.view();
	for (int i = 0; i < product(image.size()); ++i) {
		linearAccess(image_view, i) = i;
	}

	std::unique_ptr<IProblem> problem(new DoubleImageOnGpuProblem(image_view));
	testSchedulerOnSingleProblemAsync(scheduler_configuration, std::move(problem));

	for (int i = 0; i < product(image.size()); ++i) {
		BOOST_CHECK(linearAccess(image_view, i) == i * 2);
	}
}

static const std::array<SchedulerConfiguration, 7> scheduler_configurations = {
	getDefaultSchedulerConfiguration(),
	SchedulerConfiguration(SelectGpu::FIRST_GPU, ThreadsPerGpu(1)),
	SchedulerConfiguration(SelectGpu::FIRST_GPU, ThreadsPerGpu(2)),
	SchedulerConfiguration(SelectGpu::FIRST_GPU, ThreadsPerGpu(4)),
	SchedulerConfiguration(SelectGpu::ALL_GPUS, ThreadsPerGpu(1)),
	SchedulerConfiguration(SelectGpu::ALL_GPUS, ThreadsPerGpu(2)),
	SchedulerConfiguration(SelectGpu::ALL_GPUS, ThreadsPerGpu(4)),
};

/// Run on scheduler a simple problem that only sleeps for a short amount of time
BOLT_AUTO_TEST_CASE(scheduler_execution_test) {
	for(const auto& config : scheduler_configurations) {
		int num_tasks = config.getNumberOfGpus() * 5;
		testSchedulerOnSingleProblemAsync(config, makeTestProblem<SleepTask>(num_tasks));
	}
}

/// Run on scheduler a simple problem that does a few allocations on the gpu
BOLT_AUTO_TEST_CASE(scheduler_allocation_test) {
	for(const auto& config : scheduler_configurations) {
		int num_tasks = config.getNumberOfGpus() * 5;
		testSchedulerOnSingleProblemAsync(config, makeTestProblem<AllocationTask>(num_tasks));
	}
}

/// Run on gpu worker a simple sleep task
BOLT_AUTO_TEST_CASE(worker_execution_test) {
	testWorkerOnSingleTaskAsync(makeSleepTask(1024));
}

/// Run on a single scheduler multiple problems at once
BOLT_AUTO_TEST_CASE(scheduler_multiple_execution_test) {
	for(const auto& config : scheduler_configurations) {
		int num_problems = config.getNumberOfGpus() * 5;
		TestSchedulerOnMultipleProblems(config, makeListOfProblems<AllocationTask>(num_problems), true);
		TestSchedulerOnMultipleProblems(config, makeListOfProblems<AllocationTask>(num_problems), false);
	}
}

/// Run on scheduler a simple task that doubles a vector of numbers on cpu
BOLT_AUTO_TEST_CASE(double_on_cpu_test) {
	for(const auto& config : scheduler_configurations) {
		TestDoubleArrayOnCpu(config);
	}
}

/// Run on scheduler a simple task that doubles a 3D image of numbers on gpu
BOLT_AUTO_TEST_CASE(double_on_gpu_test) {
	for(const auto& config : scheduler_configurations) {
		testDoubleImageOnGpu(config);
	}
}

BOLT_AUTO_TEST_CASE(allocation_failure_test) {
	SchedulerConfiguration config = getDefaultSchedulerConfiguration();
	BOOST_REQUIRE_THROW(testSchedulerOnSingleProblemAsync(config, makeTestProblem<InfiniteAllocationTask>(1)), OutOfMemory);
	BOOST_REQUIRE_THROW(testSchedulerOnSingleProblem(config, makeTestProblem<InfiniteAllocationTask>(1)), OutOfMemory);
}

BOLT_AUTO_TEST_CASE(cuda_oversized_test) {
	SchedulerConfiguration config = getDefaultSchedulerConfiguration();
	BOOST_REQUIRE_THROW(testSchedulerOnSingleProblemAsync(config, makeTestProblem<OversizedTextureTask>(1)), OutOfMemory);
	BOOST_REQUIRE_THROW(testSchedulerOnSingleProblem(config, makeTestProblem<OversizedTextureTask>(1)), OutOfMemory);
}

BOLT_AUTO_TEST_CASE(fft_test) {
	for(const auto& config : scheduler_configurations) {
		int num_tasks = config.getNumberOfGpus() * 5;
		testSchedulerOnSingleProblemAsync(config, makeTestProblem<FftTask>(num_tasks));
	}
}

/// Test scheduler for task with shared cache
BOLT_AUTO_TEST_CASE(shared_cache_test) {
	SchedulerConfiguration config = SchedulerConfiguration(SelectGpu::FIRST_GPU, ThreadsPerGpu(1));
	testSchedulerOnSingleProblemAsync(config, std::unique_ptr<IProblem>(new SharedBufferProblem(0)));
}



}  // namespace mgt

}  // namespace bolt
