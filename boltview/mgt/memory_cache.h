// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/cuda_defines.h>
#include <boltview/exceptions.h>
#include <boltview/mgt/image_pointers.h>

#include <list>
#include <unordered_map>
#include <utility>

#include <boost/exception/info.hpp>
#include <boost/variant/variant.hpp>
#include <boost/variant/get.hpp>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{


/// This file contains various implementations of a memory cache.
/// A memory cache is able to:
///	 - store memory containers such as DeviceImagePointer from image_pointers.h
///	 - free the memory on request
///	 - return a compatible piece of memory for an image of desired type and size
/// NOT threadsafe

using SharedCacheBadSizeInfo = boost::error_info<struct SharedCacheBadSizeInfoTag, std::pair<uint64_t, uint64_t>>;

struct SharedCacheBadTypeError: BoltError {};

/// shared memory cache
///  O(1) store
///  O(1) free (oldest memory)
///  O(1) find
class SharedMemoryCache {
public:
	class GetImagePointerSize : public boost::static_visitor<uint64_t> {
	public:
		template<typename PtrType>
		uint64_t operator() (const PtrType& ptr) const {
			return ptr.size();
		}
	};

	using PtrVariant = boost::variant<DeviceImagePointer, TextureImagePointer, FftCalculatorPointer<Forward>, FftCalculatorPointer<Inverse>>;

	template<class TImageType>
	using PointerType = typename GetSmartPointerHelper<TImageType>::PointerType;

	template<typename TImageType>
	PointerType<TImageType> findCompatiblePointer(ImageDescriptor<TImageType> image_descriptor, uint64_t id) {
		auto result = image_wrappers_.find(id);
		if (result != image_wrappers_.end()) {
			if (boost::get<PointerType<TImageType>>(result->second).isCompatibleWith(image_descriptor)) {
				auto ptr = std::move(boost::get<PointerType<TImageType>>(result->second));
				image_wrappers_.erase(result);
				return ptr;
			} else {
				BOLT_THROW(
					SharedCacheBadTypeError()
						<< SharedCacheBadSizeInfo({image_descriptor.sizeInBytes(), boost::get<PointerType<TImageType>>(result->second).size()}));
			}
		} else {
			return PointerType<TImageType>();
		}
	}

	template<typename TImageType>
	void store(uint64_t id, TImageType&& image) {
		image_wrappers_.emplace(id, getSmartPointer(std::forward<TImageType>(image)));
	}

	uint64_t freeOne() {
		if (image_wrappers_.size() == 0){
			return 0;
		}
		uint64_t size = boost::apply_visitor(GetImagePointerSize(), image_wrappers_.begin()->second);
		image_wrappers_.erase(image_wrappers_.begin());
		return size;
	}

	uint64_t size() {
		uint64_t size = 0;
		for (const auto& pointer : image_wrappers_) {
			size += boost::apply_visitor(GetImagePointerSize(), pointer.second);
		}
		return size;
	}

private:
	std::unordered_multimap<uint64_t, PtrVariant> image_wrappers_;
};


/// LRU memory cache
///  O(1) store
///  O(1) free (least recently used memory)
///  O(N) find (N = number of elements of cache)
class LRUMemoryCache {
public:
	template<class TImageType>
	using PointerType = typename GetSmartPointerHelper<TImageType>::PointerType;

	template<typename TImageType>
	PointerType<TImageType> findCompatiblePointer(ImageDescriptor<TImageType> image_descriptor) {
		auto comparison_predicate = [image_descriptor](const PointerType<TImageType>& ptr){
			return ptr.isCompatibleWith(image_descriptor);
		};
		auto& list = getPointerStore<PointerType<TImageType>>();
		auto result = std::find_if(std::begin(list), std::end(list), comparison_predicate);
		if (result != std::end(list)) {
			PointerType<TImageType> pointer = std::move(*result);
			list.erase(result);
			return pointer;
		} else {
			return PointerType<TImageType>();
		}
	}

	template<typename TImageType>
	void store(TImageType&& image) {
		auto& list = getPointerStore<PointerType<TImageType>>();
		list.emplace(std::begin(list), getSmartPointer(std::forward<TImageType>(image)));
	}

	uint64_t freeOne() {
		uint64_t size = 0;
		if ((size = freeOne<DeviceImagePointer>()) > 0) {
			return size;
		}
		if ((size = freeOne<TextureImagePointer>()) > 0) {
			return size;
		}
		if ((size = freeOne<FftCalculatorPointer<Forward>>()) > 0) {
			return size;
		}
		if ((size = freeOne<FftCalculatorPointer<Inverse>>()) > 0) {
			return size;
		}
		return 0;
	}

	uint64_t size() {
		return size<DeviceImagePointer>() + size<TextureImagePointer>() + size<FftCalculatorPointer<Forward>>() + size<FftCalculatorPointer<Inverse>>();
	}

	uint64_t count() {
		return device_pointers_.size() + texture_pointers_.size() + fft_forward_pointers_.size() + fft_inverse_pointers_.size();
	}

	template<typename TPointerType>
	uint64_t size() {
		uint64_t size = 0;
		for (const TPointerType& pointer : getPointerStore<TPointerType>()) {
			size += pointer.size();
		}
		return size;
	}

private:
	template<typename TPointerType>
	uint64_t freeOne() {
		auto& list = getPointerStore<TPointerType>();
		if (list.size() == 0){
			return 0;
		}
		uint64_t size = list.back().size();
		list.pop_back();
		return size;
	}

	template<typename TPointerType>
	std::list<TPointerType>& getPointerStore();

	std::list<DeviceImagePointer> device_pointers_;
	std::list<TextureImagePointer> texture_pointers_;
	std::list<FftCalculatorPointer<Forward>> fft_forward_pointers_;
	std::list<FftCalculatorPointer<Inverse>> fft_inverse_pointers_;
};

template<>
inline std::list<DeviceImagePointer>& LRUMemoryCache::getPointerStore() {
	return device_pointers_;
}

template<>
inline std::list<TextureImagePointer>& LRUMemoryCache::getPointerStore() {
	return texture_pointers_;
}

template<>
inline std::list<FftCalculatorPointer<Forward>>& LRUMemoryCache::getPointerStore() {
	return fft_forward_pointers_;
}

template<>
inline std::list<FftCalculatorPointer<Inverse>>& LRUMemoryCache::getPointerStore() {
	return fft_inverse_pointers_;
}

/// Hash map memory cache
///  O(1) store
///  O(1) free (random)
///  O(K) find (K = number of elements of cache with the same memory size)
class HashMapMemoryCache {
public:
	template<class TImageType>
	using PointerType = typename GetSmartPointerHelper<TImageType>::PointerType;

	/// store the image pointers in a hash map with the key being the size of memory in bytes
	/// optional TODO(tom): put all information about image into key
	template<class TPointerType>
	using storage_type = std::unordered_multimap<uint64_t, TPointerType>;

	template<typename TImageType>
	PointerType<TImageType> findCompatiblePointer(ImageDescriptor<TImageType> image_descriptor) {
		uint64_t length_in_bytes_ = getImageSizeInBytes(image_descriptor);
		auto& map = getPointerStore<PointerType<TImageType>>();

		auto range = map.equal_range(length_in_bytes_);
		if (range.first != std::end(map)) {
			auto comparison_predicate = [image_descriptor](const std::pair<const uint64_t, PointerType<TImageType>>& ptr){
				return ptr.second.isCompatibleWith(image_descriptor);
			};
			auto result = std::find_if(range.first, range.second, comparison_predicate);
			if (result != range.second) {
				PointerType<TImageType> pointer = std::move(result->second);
				map.erase(result);
				return pointer;
			}
		}
		return PointerType<TImageType>();
	}

	template<typename TImageType>
	void store(TImageType&& image) {
		auto& map = getPointerStore<PointerType<TImageType>>();
		uint64_t size = calculateSizeInBytes(image);
		map.emplace(size, getSmartPointer(std::forward<TImageType>(image)));
	}

	uint64_t freeOne() {
		uint64_t size = 0;
		if ((size = freeOne<DeviceImagePointer>()) > 0) {
			return size;
		}
		if ((size = freeOne<TextureImagePointer>()) > 0) {
			return size;
		}
		return 0;
	}

	uint64_t size() {
		return size<DeviceImagePointer>() + size<TextureImagePointer>();
	}

	template<typename TPointerType>
	uint64_t size() {
		uint64_t size = 0;
		for (const auto& pointer : getPointerStore<TPointerType>()) {
			size += pointer.second.size();
		}
		return size;
	}

private:
	template<typename TPointerType>
	uint64_t freeOne() {
		auto& map = getPointerStore<TPointerType>();
		if (map.size() == 0){
			return 0;
		}
		uint64_t size = std::begin(map)->second.size();
		map.erase(std::begin(map));
		return size;
	}

	template<typename TPointerType>
	storage_type<TPointerType>& getPointerStore();

	storage_type<DeviceImagePointer> device_pointers_;
	storage_type<TextureImagePointer> texture_pointers_;
};

template<>
inline HashMapMemoryCache::storage_type<DeviceImagePointer>& HashMapMemoryCache::getPointerStore() {
	return device_pointers_;
}

template<>
inline HashMapMemoryCache::storage_type<TextureImagePointer>& HashMapMemoryCache::getPointerStore() {
	return texture_pointers_;
}

/// TODO(tom): benchmark which implementation to use
using MemoryCache = LRUMemoryCache;

/// @}

}  // namespace mgt

}  // namespace bolt
