// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

namespace bolt {

namespace mgt {

template<typename TImageType>
void GpuMemoryManager::releaseBuffer(TImageType && image) {
	std::lock_guard<std::mutex> lock(cache_mutex_);
	cache_.store(std::forward<TImageType>(image));
}


template<typename TImageType>
void GpuMemoryManager::releaseSharedBuffer(TImageType && image, uint64_t id) {
	std::lock_guard<std::mutex> lock(shared_cache_mutex_);
	shared_cache_.store(id, std::forward<TImageType>(image));
}


template<typename TImageType>
GpuBufferView<TImageType> GpuMemoryManager::getBuffer(ImageDescriptor<TImageType> image_descriptor) {
	// DebugPrint("Before Allocation");
	std::unique_lock<std::mutex> cache_lock(cache_mutex_);
	auto pointer = cache_.findCompatiblePointer(image_descriptor);
	cache_lock.unlock();

	bool cache_miss;
	TImageType image = acquireImage(std::move(pointer), image_descriptor, &cache_miss);

	GpuBufferView<TImageType> view(std::move(image), this, cache_miss);

	// DebugPrint("After Allocation");
	return view;
}

template<typename TImageType>
TImageType GpuMemoryManager::allocateImage(ImageDescriptor<TImageType> image_descriptor) {
	uint64_t released = 0;
	uint64_t size_bytes = getImageSizeInBytes(image_descriptor);
	std::unique_lock<std::mutex> cache_lock(cache_mutex_);
	while (amountOfReservableMemory() < size_bytes && (released = cache_.freeOne()) > 0) {
		occupied_memory_ -= released;
	}
	if (amountOfReservableMemory() >= size_bytes) {
		occupied_memory_ += size_bytes;
		const auto cacheCount = cache_.count();
		cache_lock.unlock();
		// NOTE(fidli): Cuda not necesarrily allocates size_bytes
		// sometimes it allocates waaaaaay to much, and subseqent allocations do not lower free memory on device
		// but with many processors per gpu this is not sufficient and GPU is out of memory, because all processes allocate greedily
		// Ways to approach:
		// 1. Hotfix is lowering the coeficient - is already at 0.8
		// 2. Catch this error and free cache, if we can - currently implemented
		// 3. introduce custom allocator, that allocates all the space in the algorithm and here just takes chunk of already allocated memory
		// this approach will then work with used caching implementation, where only truely size_bites would be occupied
		auto retries = cacheCount + 1;
		assert(retries >= 1);
		while(retries--){
			try{
				return image_descriptor.allocate();
			} catch (CudaError & e){
				// NOTE(fidli): Most likely out of memory
				// TODO(fidli): check more properly, Perhaps introduce CudaOutOfMemoryError
				cache_lock.lock();
				auto released = cache_.freeOne();
				occupied_memory_ -= released;
				cache_lock.unlock();
				bool success = released != 0;
				BOLT_DFORMAT("Memory exception caught. Trying to free one more cache item and carry on. Freed success: %1%. Originally there were %2% cached items.", success, cacheCount);
				if(success){
					// NOTE(fidli): This also resets error message (out of memory)
					cudaGetLastError();
				}else{ // NOTE(fidli): nothing cached, no memory freed, this would be infinite loop
					BOLT_THROW(OutOfMemory());
				}
			}
		}
		BOLT_THROW(OutOfMemory());
	} else {
		BOLT_THROW(OutOfMemory());
	}
}

template<typename TImageType, typename TFunctor>
GpuBufferView<TImageType> GpuMemoryManager::getSharedBuffer(
	ImageDescriptor<TImageType> image_descriptor,
	uint64_t id,
	boost::optional<TFunctor> before_release)
{
	std::unique_lock<std::mutex> cache_lock(shared_cache_mutex_);
	auto pointer = shared_cache_.findCompatiblePointer(image_descriptor, id);
	cache_lock.unlock();

	bool cache_miss;
	TImageType image = acquireImage(std::move(pointer), image_descriptor, &cache_miss);

	GpuBufferView<TImageType> view(std::move(image), this, cache_miss, id);

	return view;
}


template<typename TImageType>
TImageType GpuMemoryManager::acquireImage(PointerType<TImageType> &&pointer, ImageDescriptor<TImageType> image_descriptor, bool *cache_miss) {
	*cache_miss = pointer.get() == nullptr;
	if (*cache_miss) {
		return allocateImage(image_descriptor);
	} else {
		return getImage(std::move(pointer), image_descriptor);
	}
}


}  // namespace mgt

}  // namespace bolt
