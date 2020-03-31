// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#if !defined(__CUDACC__)
	#error gpu_memory_manager.h cannot be included in file which is not compiled by nvcc!
#endif

#include <algorithm>
#include <functional>
#include <list>
#include <mutex>
#include <numeric>
#include <string>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/optional.hpp>
#include <boost/range/irange.hpp>

#include <boltview/mgt/device_code.h>
#include <boltview/mgt/memory_cache.h>
#include <boltview/mgt/gpu_worker_configuration.h>
#include <boltview/fft/fft_utils.h>


#include <boltview/int_sequence.h>
#include <boltview/variadic_templates.h>




namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{


struct AllocationError: MgtError {};

struct ExecutionError: MgtError {};

struct OutOfMemory: AllocationError {};

struct TaskTooBig: AllocationError {};

template<typename TImageType>
class GpuBufferView;

/// GpuMemoryManager allocates gpu buffers on a single assigned gpu.
/// Lifetime of the gpu buffers is managed by the GpuBufferView object.
/// Instead of immediately freeing the images, their memory is stored in a
/// cache and the memory manager then uses it to fulfill subsequent requests.
/// TODO(tom) features:
///  - buffer allocation via a unique id - allows sharing data between tasks
///  - custom callbacks on deallocation - allows lazy copy out to main memory
class GpuMemoryManager {
public:
	template<typename TImageType>
	using PointerType = typename GetSmartPointerHelper<TImageType>::PointerType;
	/// Allocates an image of specified type and size,
	/// an owning view to the image is returned.
	/// \threadsafe
	/// \param image_descriptor
	template<typename TImageType>
	GpuBufferView<TImageType> getBuffer(ImageDescriptor<TImageType> image_descriptor);

	/// Return gpu buffer view of a specific buffer given by an id number
	/// consecutive tasks can call this function with same size and id and they will get a
	/// view to the same memory.
	/// \threadsafe
	/// \param image_descriptor
	/// \param id unique identifier for the requested buffer
	/// \param before_release functor that will be called just before release of this buffer
	template<typename TImageType, typename TFunctor = std::function<void()>>
	GpuBufferView<TImageType> getSharedBuffer(
			ImageDescriptor<TImageType> image_descriptor,
			uint64_t id,
			boost::optional<TFunctor> before_release = boost::optional<std::function<void()>>());

	/// Releases target image and moves its memory to cache.
	/// \threadsafe
	/// \param image the image that is to be released
	template<typename TImageType>
	void releaseBuffer(TImageType && image);


	/// Releases target image and moves its memory to cache.
	/// \threadsafe
	/// \param image the image that is to be released
	template<typename TImageType>
	void releaseSharedBuffer(TImageType && image, uint64_t id);


	void setGpu(int gpu_id, uint64_t available_memory) {
		device::setCudaDevice(gpu_id);
		total_memory_ = available_memory;
		occupied_memory_ = 0;
	}

	void debugPrint(std::string message) const {
		BOLT_DFORMAT("%1%, %2%%% occupied", message, 100 * occupied_memory_ / double(total_memory_));
	}

private:
	template<typename TImageType>
	TImageType acquireImage(PointerType<TImageType> &&pointer, ImageDescriptor<TImageType> image_descriptor, bool *cache_miss);

	template<typename TImageType>
	TImageType allocateImage(ImageDescriptor<TImageType> image_descriptor);

	uint64_t amountOfReservableMemory() const {
		return total_memory_ - occupied_memory_;
	}

	std::mutex cache_mutex_;
	std::mutex shared_cache_mutex_;
	std::mutex image_store_mutex_;

	uint64_t total_memory_;
	uint64_t occupied_memory_;
	MemoryCache cache_;
	SharedMemoryCache shared_cache_;
};

/// GpuBufferView is the return type of GpuMemoryManager methods,
/// it allows access to on demand allocated gpu buffers.
template<typename TImageType>
class GpuBufferView {
public:
	using ImageType = TImageType;

	GpuBufferView() : call_(nullptr) {}
	explicit GpuBufferView(ImageType && image, GpuMemoryManager* call, bool fresh = true, boost::optional<uint64_t> id = boost::optional<uint64_t>{}) :
		image_(std::forward<ImageType>(image)),
		call_(call),
		fresh_(fresh),
		id_(id)
	{}

	~GpuBufferView() {
		deallocate();
	}

	GpuBufferView(GpuBufferView&&) = default;
	GpuBufferView& operator=(GpuBufferView&&) = default;

	ImageType* operator->() {
		return &image_;
	}

	const ImageType* operator->() const{
		return &image_;
	}

	ImageType & Reference() {
		return image_;
	}


	/// Record that the buffer is free to use again.
	void release() {
		deallocate();
		call_.reset(nullptr);
	}

	/// Returns true if the call to GpuMemoryManager that produced this buffer had to allocate memory.
	/// This is used to decide if to initiate copy from cpu memory for shared buffers.
	bool isFresh() const {
		return fresh_;
	}

private:
	void deallocate() {
		if (call_.get()) {
			if (id_) {
				call_->releaseSharedBuffer(std::move(image_), *id_);
			} else {
				call_->releaseBuffer(std::move(image_));
			}
		}
	}

	struct GpuMemoryReleaser {
		void operator()(GpuMemoryManager* ptr) {}
	};

	bool fresh_;
	ImageType image_;
	std::unique_ptr<GpuMemoryManager, GpuMemoryReleaser> call_;
	boost::optional<uint64_t> id_;
};


/// Utility class wraps several buffer types defined as template parameters.
/// It can allocate and estimate memory occupied in GpuMemoryManager, based on the parameters which would be passed to GpuMemoryManager.
/// If the respectve buffer allocation would require more than one parameter (e.g. FFTCalculator) these must be bundled in a tuple.
template<typename ...TBufferTypes>
class BufferAllocationWrapper {
public:
	using ManagedBuffers = std::tuple<bolt::mgt::GpuBufferView<TBufferTypes>...>;

	template<typename ...TAllocationArgs>
	void Allocate(bolt::mgt::GpuMemoryManager* gpu, const TAllocationArgs & ... arguments) {
		static_assert(sizeof...(TBufferTypes) == sizeof...(TAllocationArgs), "Argument count is different from the buffer count");
		AllocateImpl(gpu, std::make_tuple(arguments...), bolt::MakeIntSequence<sizeof...(TAllocationArgs)>{});
	}

	template<typename ...TAllocationArgs>
	static int64_t Estimate(const TAllocationArgs & ... arguments) {
		static_assert(sizeof...(TBufferTypes) == sizeof...(TAllocationArgs), "Argument count is different from the buffer count");
		return EstimateImpl(std::make_tuple(arguments...), bolt::MakeIntSequence<sizeof...(TAllocationArgs)>{});
	}

	ManagedBuffers &buffers () {
		return buffers_;
	}

private:
	template <typename TTuple, int... tI>
	void AllocateImpl(bolt::mgt::GpuMemoryManager* gpu, const TTuple &arguments, bolt::IntSequence<tI...>) {
		std::initializer_list<int> {
			(this->AllocateOneImpl<
				typename std::tuple_element<tI, TTuple>::type,
				typename bolt::Index<tI, TBufferTypes...>::type,
				tI>(gpu, std::get<tI>(arguments)), 0)... };
	}

	template<typename TArgument, typename TBuffer, int tIndex>
	void AllocateOneImpl(bolt::mgt::GpuMemoryManager* gpu, const TArgument &argument) {
		AllocateOneHelper<tIndex, TBuffer>::Call(&buffers_, gpu, argument);
	}

	template <typename TTuple, int... tI>
	static int64_t EstimateImpl(const TTuple &arguments, bolt::IntSequence<tI...>) {
		auto sizes = std::initializer_list<int64_t> {
			EstimateOneImpl<
				typename std::tuple_element<tI, TTuple>::type,
				typename bolt::Index<tI, TBufferTypes...>::type,
				tI>(std::get<tI>(arguments))... };
		return std::accumulate(begin(sizes), end(sizes), int64_t(0));
	}

	template<typename TArgument, typename TBuffer, int tIndex>
	static int64_t EstimateOneImpl(const TArgument &argument) {
		return AllocateOneHelper<tIndex, TBuffer>::Estimate(argument);
	}

	template<int tIndex, typename TBuffer>
	struct AllocateOneHelper {

		template<typename TBuffers, typename TArgument>
		static void Call(TBuffers *buffers, bolt::mgt::GpuMemoryManager* gpu, const TArgument &argument) {
			bolt::mgt::ImageDescriptor<TBuffer> descriptor(argument);
			std::get<tIndex>(*buffers) = gpu->getBuffer(descriptor);
		}

		template<typename TArgument>
		static int64_t Estimate(const TArgument &argument) {
			return bolt::mgt::getImageSizeInBytesImpl<typename TBuffer::Element>(argument);
		}
	};

	template<int tIndex, int tDim, typename TPolicy>
	struct AllocateOneHelper<tIndex, bolt::FftCalculator<tDim, TPolicy>> {
		template<typename TBuffers, typename ...TArguments>
		static void Call(TBuffers *buffers, bolt::mgt::GpuMemoryManager* gpu, const std::tuple<TArguments...> &arguments) {
			bolt::mgt::ImageDescriptor<bolt::FftCalculator<tDim, TPolicy>> descriptor(arguments);
			std::get<tIndex>(*buffers) = gpu->getBuffer(descriptor);
		}

		template<typename ...TArguments>
		static int64_t Estimate(const std::tuple<TArguments...> &arguments) {
			return bolt::FftCalculator<tDim, TPolicy>::EstimateWorkArea(arguments);
		}


	};

	std::tuple<bolt::mgt::GpuBufferView<TBufferTypes>...> buffers_;
};

/// Return maximal batch size, which would fit into the provided memory limit.
/// \param first size of smallest possible batch
/// \param last size of largest possible batch
/// \param available_memory memory in bytes
/// \param size_estimator estimator function, which computes memory required by tested batch size
/// \return pair containing chosen batch size and amount of memory it would need. If no such batch exists, it returns {-1, -1}.
inline std::pair<int, int64_t> getMaximalPossibleBatchSize(int first, int last, int64_t available_memory, std::function<int64_t(int)> size_estimator) {
	// TODO(johny) can be noexcept?
	if (first > last) {
		return {0, 0};
	}
	auto range = boost::irange(first, last + 1);
	auto start = boost::make_transform_iterator(range.begin(), size_estimator);
	auto end = boost::make_transform_iterator(range.end(), size_estimator);

	auto found = std::upper_bound(start, end, available_memory);

	if (found == start){
		return { -1, -1};
	}
	// correct batch size is the one before the upper limit
	--found;
	int batch_size = *(found.base());
	return { batch_size, *found };
}


/// @}


}  // namespace mgt

}  // namespace bolt

#include "gpu_memory_manager.tcc"
