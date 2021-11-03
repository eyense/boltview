#define BOOST_TEST_MODULE SynchronizationTest

#include <boltview/device_future.h>
#include <boltview/stream.h>

#include <boltview/copy.h>
#include <boltview/for_each.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/image_io.h>

#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boltview/tests/test_utils.h>

#include <chrono>
#include <thread>
#include <stdexcept>


namespace bolt {


typedef long long clock_value_t;
// gpu "sleep" function
__global__ void sleep(clock_value_t sleep_cycles) {
	clock_value_t start = clock64();
	clock_value_t cycles_elapsed;
	do {
		cycles_elapsed = clock64() - start;
	} while (cycles_elapsed < sleep_cycles);
};

static int getFrequencyOfZeroDevice() {
	cudaDeviceProp prop;
	BOLT_CHECK(cudaGetDeviceProperties(&prop, 0))
	return prop.clockRate;
};
// in kilohertz
static const int kFrequency = getFrequencyOfZeroDevice();


// example of how the DeviceFutures could be returned from async operations on gpu
// The result setter is invoked only after the sleep kernel
DeviceFuture<int> computationWithResultAsync(cudaStream_t stream, int result, std::shared_ptr<bool> thread_tracker) {
	sleep<<<1, 1, 0, stream>>>(500 * kFrequency);
	auto resultSetter = [=]() {
		thread_tracker.use_count();
		return result;
	};
	return DeviceFuture<int>(stream, resultSetter);
}

DeviceFuture<void> computationAsync(cudaStream_t stream, std::shared_ptr<bool> thread_tracker) {
	sleep<<<1, 1, 0, stream>>>(500 * kFrequency);
	return DeviceFuture<void>(stream, [=]() { thread_tracker.use_count(); });
}

DeviceFuture<void> computationWithSetterExceptionAsync(cudaStream_t stream, std::shared_ptr<bool> thread_tracker) {
	sleep<<<1, 1, 0, stream>>>(500 * kFrequency);
	return DeviceFuture<void>(stream, [=]() {
		thread_tracker.use_count();
		throw std::runtime_error("error");
	});
}

BOLT_AUTO_TEST_CASE(DeviceFutureWithValuesTest) {
	CudaStream stream;
	std::vector<int> checker;
	std::shared_ptr<bool> thread_tracker = std::make_shared<bool>(true);
	{
		// binds the shared pointer thread_tracker to every lambda that is executed in a detached thread
		// as long as the valueHolder object exists, the shared_ptr is bound to it, having inceremented use_count
		auto f = computationWithResultAsync(stream.get(), 0, thread_tracker)
					 .Then([&, thread_tracker](int i) {
						 thread_tracker.use_count();
						 checker.push_back(i);
						 return 1;
					 })
					 .Then([&, thread_tracker](int i) {
						 thread_tracker.use_count();
						 checker.push_back(i);
						 return computationWithResultAsync(stream.get(), 2, thread_tracker).getValue();
					 })
					 .Then([&, thread_tracker](int i) {
						 thread_tracker.use_count();
						 checker.push_back(i);
					 });

		f.wait();
		BOOST_CHECK_EQUAL(checker.size(), 3);
		for (int i = 0; i < checker.size(); i++) {
			BOOST_CHECK_EQUAL(checker[i], i);
		}
	}
	// All valueHolder objects should be destructed by now, os the thread_tracker is bound only to this function context
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	BOOST_CHECK_EQUAL(thread_tracker.use_count(), 1);
};

BOLT_AUTO_TEST_CASE(DeviceFutureWithVoidTest) {
	CudaStream stream;
	std::vector<int> checker;
	std::shared_ptr<bool> thread_tracker = std::make_shared<bool>(true);
	{
		auto f = computationAsync(stream.get(), thread_tracker)
					 .Then([&, thread_tracker]() {
						 thread_tracker.use_count();
						 checker.push_back(0);
						 return computationAsync(stream.get(), thread_tracker);
					 })
					 .Then([&, thread_tracker](DeviceFuture<void> previousTask) {
						 thread_tracker.use_count();
						 previousTask.wait();
						 checker.push_back(1);
					 });

		f.wait();

		BOOST_CHECK_EQUAL(checker.size(), 2);
		BOOST_CHECK_EQUAL(checker[0], 0);
		BOOST_CHECK_EQUAL(checker[1], 1);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	BOOST_CHECK_EQUAL(thread_tracker.use_count(), 1);
};


bool checkError(const std::runtime_error& err) {
	BOOST_CHECK_EQUAL(err.what(), "error");
	return true;
}

// Exception is propagated into the promise
BOLT_AUTO_TEST_CASE(ExceptionInSetterTest) {
	CudaStream stream;
	std::vector<int> checker;
	std::shared_ptr<bool> thread_tracker = std::make_shared<bool>(true);
	{
		auto f = computationWithSetterExceptionAsync(stream.get(), thread_tracker);
		// Result setter throws exception which is catched and stored in the promise object
		f.wait();  // wait on the future actually doesnt rethrow exception,
		// only waits for the exception to propagate to the promise
		BOOST_CHECK_EXCEPTION(f.getValue(), std::runtime_error, checkError);  // getValue rethrows the exception
	}
	// Verify that the thread has destroyed ValueHolder object, and finished
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	BOOST_CHECK_EQUAL(thread_tracker.use_count(), 1);
};

BOLT_AUTO_TEST_CASE(ExceptionPropagationTest) {
  	//Exception is propagated and stored in the promise chain
	CudaStream stream;
	std::shared_ptr<bool> thread_tracker = std::make_shared<bool>(true);
	{
		auto f = computationAsync(stream.get(), thread_tracker)
					 .Then([&, thread_tracker]() {
						 computationWithSetterExceptionAsync(stream.get(), thread_tracker).getValue();
						 return 1;
					 })
					 .Then([&, thread_tracker](int i) {
  	  	  	  	  	  	 computationAsync(stream.get(), thread_tracker).getValue();
						 thread_tracker.use_count();
						 return i;
					 });

		BOOST_CHECK_EXCEPTION(f.getValue(), std::runtime_error, checkError);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	BOOST_CHECK_EQUAL(thread_tracker.use_count(), 1);
};


BOLT_AUTO_TEST_CASE(DeviceFutureMultipleBranchesTest) {
	CudaStream stream;
	std::vector<int> checker1;
	std::vector<int> checker2;
	std::shared_ptr<bool> thread_tracker = std::make_shared<bool>(true);
	{
		auto f = computationAsync(stream.get(), thread_tracker)
					 .Then([&, thread_tracker]() {
						 checker1.push_back(0);
						 checker2.push_back(0);
						 thread_tracker.use_count();
						 return computationAsync(stream.get(), thread_tracker);
					 })
					 .Then([&, thread_tracker](DeviceFuture<void> previousTask) {
						 thread_tracker.use_count();
						 previousTask.wait();
						 checker1.push_back(1);
						 checker2.push_back(1);
					 });

		auto g = f.Then([&, thread_tracker]() {
					  checker1.push_back(2);
					  thread_tracker.use_count();
					  return computationWithResultAsync(stream.get(), 3, thread_tracker).getValue();
				  })
					 .Then([&, thread_tracker](int i) {
						 checker1.push_back(i);
						 thread_tracker.use_count();
						 return 4;
					 })
					 .Then([&, thread_tracker](int i) {
						 thread_tracker.use_count();
						 checker1.push_back(i);
					 });

		auto h = f.Then([&, thread_tracker]() {
			thread_tracker.use_count();
			checker2.push_back(2);
		});

		h.wait();
		g.wait();

		BOOST_CHECK_EQUAL(checker1.size(), 5);
		for (int i = 0; i < checker1.size(); i++) {
			BOOST_CHECK_EQUAL(checker1[i], i);
		}


		BOOST_CHECK_EQUAL(checker2.size(), 3);
		for (int i = 0; i < checker2.size(); i++) {
			BOOST_CHECK_EQUAL(checker2[i], i);
		}
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	BOOST_CHECK_EQUAL(thread_tracker.use_count(), 1);
};

}  // namespace bolt
