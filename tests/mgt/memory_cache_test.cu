// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#define BOOST_TEST_MODULE MemoryCacheTest
#include <boost/test/included/unit_test.hpp>

#include <boltview/math/vector.h>
#include <boltview/device_image.h>
#include <boltview/mgt/image_pointers.h>
#include <boltview/mgt/memory_cache.h>
#include <boltview/texture_image.h>

#include "../test_utils.h"

namespace bolt {

namespace mgt {

/// Store image memory based on TStoreImageDescriptor into cache, then try to find
/// memory for an image based on TFindImageDescriptor in the cache.
/// Check of result is dependent on value of expect_find.
template<typename TMemoryCache, typename TStoreImageDescriptor, typename TFindImageDescriptor>
void TestCacheStoreAndFind(
	TStoreImageDescriptor store_image_desc,
	TFindImageDescriptor find_image_desc,
	bool expect_find = true)
{
	TMemoryCache memory_cache;
	typename TStoreImageDescriptor::ImageType image(store_image_desc.size());
	uint64_t size = calculateSizeInBytes(image);

	memory_cache.store(std::move(image));
	BOOST_CHECK(memory_cache.size() == size);

	auto new_pointer = memory_cache.findCompatiblePointer(find_image_desc);

	if (expect_find == true) {
		BOOST_CHECK(new_pointer.get() != nullptr);
		BOOST_CHECK(memory_cache.size() == 0);
	} else {
		BOOST_CHECK(new_pointer.get() == nullptr);
		BOOST_CHECK(memory_cache.size() == size);
	}
}


template<typename TMemoryCache, typename TStoreImageDescriptor, typename TFindImageDescriptor>
void TestSharedCacheStoreAndFind(
	TStoreImageDescriptor store_image_desc,
	TFindImageDescriptor find_image_desc,
	bool expect_find,
	uint64_t store_id,
	uint64_t find_id)
{
	TMemoryCache memory_cache;
	typename TStoreImageDescriptor::ImageType image(store_image_desc.size());
	uint64_t size = calculateSizeInBytes(image);

	memory_cache.store(store_id, std::move(image));
	BOOST_CHECK(memory_cache.size() == size);

	auto new_pointer = memory_cache.findCompatiblePointer(find_image_desc, find_id);

	if (expect_find == true) {
		BOOST_CHECK(new_pointer.get() != nullptr);
		BOOST_CHECK(memory_cache.size() == 0);
	} else {
		BOOST_CHECK(new_pointer.get() == nullptr);
		BOOST_CHECK(memory_cache.size() == size);
	}
}


class TestCacheStoreAndFindSingleImage {
public:
	template<typename TMemoryCache, typename TImage>
	void test() {
		typename TImage::SizeType size = TImage::SizeType::fill(100);
		TestCacheStoreAndFind<TMemoryCache>(ImageDescriptor<TImage>(size), ImageDescriptor<TImage>(size));
	}
};


class TestSharedCacheStoreAndFindSingleImage {
public:
	template<typename TMemoryCache, typename TImage>
	void test() {
		typename TImage::SizeType size1 = TImage::SizeType::fill(100);
		typename TImage::SizeType size2 = TImage::SizeType::fill(50);
		TestSharedCacheStoreAndFind<TMemoryCache>(ImageDescriptor<TImage>(size1), ImageDescriptor<TImage>(size1), true, 1, 1);
		TestSharedCacheStoreAndFind<TMemoryCache>(ImageDescriptor<TImage>(size1), ImageDescriptor<TImage>(size1), false, 1, 2);
		BOOST_CHECK_THROW(
			TestSharedCacheStoreAndFind<TMemoryCache>(ImageDescriptor<TImage>(size1), ImageDescriptor<TImage>(size2), false, 1, 1),
			SharedCacheBadTypeError);
	}
};


/// Store and free multiple images.
template<typename TMemoryCache, typename TStoreImageDescriptor>
void TestCacheStoreFree(TStoreImageDescriptor store_image_desc, int number_of_images_to_store) {
	TMemoryCache memory_cache;
	uint64_t size = getImageSizeInBytes(store_image_desc);

	for (int i = 1; i <= number_of_images_to_store; i++) {
		typename TStoreImageDescriptor::ImageType image(store_image_desc.size());
		memory_cache.store(std::move(image));
		BOOST_CHECK(memory_cache.size() == size * i);
	}

	for (int i = 1; i <= number_of_images_to_store; i++) {
		BOOST_CHECK(memory_cache.freeOne() == size);
		BOOST_CHECK(memory_cache.size() == size * (number_of_images_to_store - i));
	}
}


template<typename TMemoryCache, typename TStoreImageDescriptor>
void TestSharedCacheStoreFree(TStoreImageDescriptor store_image_desc, int number_of_images_to_store, uint64_t id) {
	TMemoryCache memory_cache;
	uint64_t size = getImageSizeInBytes(store_image_desc);

	for (int i = 1; i <= number_of_images_to_store; i++) {
		typename TStoreImageDescriptor::ImageType image(store_image_desc.size());
		memory_cache.store(id, std::move(image));
		BOOST_CHECK(memory_cache.size() == size * i);
	}

	for (int i = 1; i <= number_of_images_to_store; i++) {
		BOOST_CHECK(memory_cache.freeOne() == size);
		BOOST_CHECK(memory_cache.size() == size * (number_of_images_to_store - i));
	}
}


class TestCacheStoreAndFreeMultipleImages {
public:
	template<typename TMemoryCache, typename TImage>
	void test() {
		typename TImage::SizeType size = TImage::SizeType::fill(100);
		TestCacheStoreFree<TMemoryCache>(ImageDescriptor<TImage>(size), 10);
	}
};

class TestSharedCacheStoreAndFreeMultipleImages {
public:
	template<typename TMemoryCache, typename TImage>
	void test() {
		typename TImage::SizeType size = TImage::SizeType::fill(100);
		TestSharedCacheStoreFree<TMemoryCache>(ImageDescriptor<TImage>(size), 10, 1);
	}
};

template<typename TFunc, typename TMemoryCache, typename TElementType>
void GenerateCacheTypeTests() {
	TFunc().template test<TMemoryCache, DeviceImage<TElementType, 2>>();
	TFunc().template test<TMemoryCache, DeviceImage<TElementType, 3>>();
	TFunc().template test<TMemoryCache, TextureImage<TElementType, 2>>();
	TFunc().template test<TMemoryCache, TextureImage<TElementType, 3>>();
}

template<typename TFunc, typename TMemoryCache>
void GenerateCacheTests() {
	GenerateCacheTypeTests<TFunc, TMemoryCache, int>();
	GenerateCacheTypeTests<TFunc, TMemoryCache, float>();
	GenerateCacheTypeTests<TFunc, TMemoryCache, Int2>();
	GenerateCacheTypeTests<TFunc, TMemoryCache, Float2>();
	GenerateCacheTypeTests<TFunc, TMemoryCache, Int4>();
	GenerateCacheTypeTests<TFunc, TMemoryCache, Float4>();
}

template<typename TFunc>
void GenerateTests() {
	GenerateCacheTests<TFunc, LRUMemoryCache>();
	GenerateCacheTests<TFunc, HashMapMemoryCache>();
}


template<typename TFunc>
void GenerateSharedCacheTests() {
	GenerateCacheTests<TFunc, SharedMemoryCache>();
}

/// Test store/find of a single image type. For all combinations of cache/image
BOLT_AUTO_TEST_CASE(cache_store_find_test) {
	GenerateTests<TestCacheStoreAndFindSingleImage>();
	GenerateSharedCacheTests<TestSharedCacheStoreAndFindSingleImage>();
}

/// Test store/free of a multiple images. For all combinations of cache/image
BOLT_AUTO_TEST_CASE(cache_store_free_test) {
	GenerateTests<TestCacheStoreAndFreeMultipleImages>();
	GenerateSharedCacheTests<TestSharedCacheStoreAndFreeMultipleImages>();
}

/// Helper class for constructing device / texture descriptors
template<typename TElement, int tDim>
class TestDescriptor {
public:
	typedef Vector<int, tDim> SizeType;
	typedef DeviceImageDescriptor<TElement, tDim> Device;
	typedef TextureImageDescriptor<TElement, tDim> Texture;

	explicit TestDescriptor(SizeType size) : size_(size) {}

	Device getDeviceImageDescriptor() {
		return Device(size_);
	}

	Texture getTextureImageDescriptor() {
		return Texture(size_);
	}

private:
	SizeType size_;
};

template<typename TMemoryCache, typename TStoreDescriptor, typename TFindDescriptor>
void TestStoreFind(TStoreDescriptor store, TFindDescriptor find, bool expect_find_device, bool expect_find_texture) {
	TestCacheStoreAndFind<TMemoryCache>(store.getDeviceImageDescriptor(), find.getDeviceImageDescriptor(), expect_find_device);
	TestCacheStoreAndFind<TMemoryCache>(store.getTextureImageDescriptor(), find.getTextureImageDescriptor(), expect_find_texture);
}

/// Memory can be reused between different types of images, this is tested here.
template<typename TMemoryCache>
void TestStoreFindCompatibility() {
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 50, 100)), TestDescriptor<float, 3>(Int3(100, 100, 50)), true, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(50, 100, 100)), TestDescriptor<float, 3>(Int3(100, 50, 100)), true, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 100, 100)), TestDescriptor<int, 3>(Int3(100, 100, 100)), true, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 100, 100)), TestDescriptor<int, 2>(Int2(10000, 100)), true, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 100, 100)), TestDescriptor<Float2, 3>(Int3(100, 100, 50)), true, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 100, 100)), TestDescriptor<float, 3>(Int3(100, 100, 50)), false, false);
	TestStoreFind<TMemoryCache>(TestDescriptor<float, 3>(Int3(100, 100, 100)), TestDescriptor<float, 2>(Int2(100, 100)), false, false);
}

/// Test store/find in cache for combinations of image types.
BOLT_AUTO_TEST_CASE(cache_store_find_compatibility_test) {
	TestStoreFindCompatibility<LRUMemoryCache>();
	TestStoreFindCompatibility<HashMapMemoryCache>();
}

template<typename TMemoryCache>
void TestFree() {
	TMemoryCache memory_cache;
	DeviceImage<float, 3> image(Int3(100, 100, 100));
	float* ptr = image.pointer();

	memory_cache.store(std::move(image));
	memory_cache.freeOne();

	// second free on a pointer should fail
	BOOST_CHECK(cudaFree(ptr) != cudaSuccess);
}


template<typename TMemoryCache>
void TestSharedCacheFree() {
	TMemoryCache memory_cache;
	DeviceImage<float, 3> image(Int3(100, 100, 100));
	float* ptr = image.pointer();

	memory_cache.store(1, std::move(image));
	memory_cache.freeOne();

	// second free on a pointer should fail
	BOOST_CHECK(cudaFree(ptr) != cudaSuccess);
}


/// Test that memory cache frees acquired memory
BOLT_AUTO_TEST_CASE(cache_free_test) {
	TestFree<LRUMemoryCache>();
	TestFree<HashMapMemoryCache>();
	TestSharedCacheFree<SharedMemoryCache>();
}

}  // namespace mgt

}  // namespace bolt
