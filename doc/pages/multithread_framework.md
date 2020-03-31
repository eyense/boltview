# Multithreaded GPU Scheduling Framework {#multithread_framework}


```
namespace mgt = bolt::mgt;

// Task is a part of a problem that fits on a single GPU
class SquareTask : public mgt::BaseTask {
public:
	SquareTask(bolt::HostImageView<float, 3> input_view, mgt::ITask::GroupId task_group_id)
		: BaseTask(task_group_id, 0), input_view_(input_view)
	{}

	void allocate(mgt::GpuMemoryManager* gpu) {
		device_image_ = gpu->getBuffer(mgt::DeviceImageDescriptor<float, 3>(input_view_.size()));
	}

	void wxecute(cudaStream_t stream) {
		auto device_view = view(device_image_);

		// Copy input onto device, compute and copy back
		bolt::copyAsync(input_view_, device_view, stream);
		transform(device_view, device_view, bolt::SquareFunctor(), stream);
		bolt::copyAsync(device_view, input_view_, stream);

		BOLT_CHECK(cudaStreamSynchronize(stream));
	}

private:
	bolt::HostImageView<float, 3> input_view_;
	mgt::GpuBufferView<bolt::DeviceImage<float, 3>> device_image_;
};

// Problem defines some work that needs to be done on one or more gpus. It splits
// into multiple tasks, each of which can be run on a single gpu
class SquareProblem : public mgt::BaseProblem {
public:
	SquareProblem(bolt::HostImageView<float, 3> input_view, mgt::IProblem::ProblemId problem_id = 0)
	 	: BaseProblem(problem_id), input_view_(input_view)
	{}

	// Create and return individual tasks
	// In this case, split input view in yz plane according to available memory
	std::vector<std::unique_ptr<mgt::ITask>> generateTasks(mgt::ResourceConstraints resource_constraints) {
		auto available_memory = resource_constraints.getLeastCapableDeviceProperties().total_memory; // GPU memory available for 1 thread
		std::vector<std::unique_ptr<mgt::ITask>> tasks;

		auto smallest_chunk_size_bytes = input_view_.size()[1] * input_view_.size()[2] * sizeof(float);

		int max_chunk_size = std::min(static_cast<int>(available_memory / smallest_chunk_size_bytes), input_view_.size()[0]);
		int number_of_tasks = std::ceil(static_cast<float>(input_view_.size()[0]) / max_chunk_size);

		for (int i = 0; i < number_of_tasks; ++i) {
			int chunk_start = i * max_chunk_size;
			int chunk_end = std::min(chunk_start + max_chunk_size, input_view_.size()[0]);
			int actual_chunk_size = chunk_end - chunk_start;

			bolt::Region<3> subview_region;
			subview_region.corner = {chunk_start, 0, 0};
			subview_region.size = {actual_chunk_size, input_view_.size()[1], input_view_.size()[2]};

			auto task_subview = bolt::subview(input_view_, subview_region);
			tasks.emplace_back(new SquareTask(task_subview, i));
		}

		return tasks;
	}

private:
	bolt::HostImageView<float, 3> input_view_;
};

void RunMultithread() {
	bolt::HostImage<float, 3> image(100, 100, 100);
	auto imview = bolt::view(image);
	bolt::fill(imview, 2.0f);

	std::unique_ptr<mgt::IProblem> problem(new SquareProblem(imview));

	int rank = 0;
	int num_ranks = 1;
	int available_gpus = 1;

	mgt::SchedulerConfiguration scheduler_configuration
		= mgt::getSchedulerConfiguration(rank, num_ranks, bolt::mgt::ThreadsPerGpu(4), mgt::UseGpuMemoryFraction(0.8f), available_gpus);

	mgt::Scheduler scheduler(scheduler_configuration);
	scheduler.addProblem(std::move(problem));
	scheduler.runUntilFinished();
}
```
