// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE AlgorithmTest
#include <boost/test/included/unit_test.hpp>

#include <boltview/array_view.h>
#include <boltview/copy.h>
#include <boltview/for_each.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/image_locator.h>
#include <boltview/image_view_utils.h>
#include <boltview/reduce.h>
#include <boltview/transform.h>
#include <boltview/convolution.h>
#include <boltview/image_io.h>

#ifdef BOLT_USE_UNIFIED_MEMORY
#include <boltview/unified_image.h>
#endif

#include "test_defs.h"
#include "test_utils.h"

namespace bolt {

struct CopyLocatorFunctor {
	template<typename TLocator>
	BOLT_DECL_HYBRID typename TLocator::Element
	operator()(TLocator locator) const {
		return locator.get();
	}
};

BOLT_AUTO_TEST_CASE(TransformLocatorCopyThroughLocatorWithSharedMemoryAndOverlap) {
	Int2 size(10,10);

	HostImage<float, 2> image(size);
	HostImage<float, 2> image_out_shared(size);

	DeviceImage<float, 2> device_image(size);
	DeviceImage<float, 2> device_image_out_shared(size);

	auto device_view = device_image.view();
	auto device_view_out_shared = device_image_out_shared.view();

	for (int i = 0; i < product(size); ++i){
		linearAccess(image.view(), i) = i;
	}
	copy(image.view(), device_view);

	auto preloadPolicy = getDefaultConvolutionPolicy<decltype(device_image.constView()), decltype(device_view_out_shared), BorderHandling::kRepeat, true>(device_view, device_view_out_shared);

	Vector<int, 2> kernel_size(3,3);
	Vector<int, 2> kernel_center(1,1);
	preloadPolicy.setPreload(kernel_size, kernel_center);

	transformLocator(
		device_image.constView(),
		device_view_out_shared,
		CopyLocatorFunctor(),
		preloadPolicy
	);

	copy(device_view_out_shared, image_out_shared.view());

	for (int i = 0; i < product(size); ++i){
		BOOST_CHECK_EQUAL(linearAccess(image.view(), i), linearAccess(image_out_shared.view(), i));
	}

	BOLT_CHECK_ERROR_STATE("Shared Memory Convolution");
}

BOLT_AUTO_TEST_CASE(TransformLocatorCopyThroughLocator) {
	HostImage<int, 2> host_image(8, 8);
	HostImage<int, 2> output_image(8, 8);
	auto checker_board = checkerboard(1, 0, Int2(2, 2), Int2(8, 8));
	copy(checker_board, host_image.view());

	auto view = host_image.constView();
	auto output_view = host_image.view();

	transformLocator(view, output_view, CopyLocatorFunctor());

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			BOOST_CHECK_EQUAL(view[Int2(i,j)], output_view[Int2(i,j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DeviceTransform) {
	DeviceImage<int, 2> result_image(16, 16);
	auto checker_board = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));

	transform(checker_board, result_image.view(), []__device__ __host__(int element){return element;});

	BOOST_CHECK_EQUAL(sumSquareDifferences(result_image.constView(), checker_board, 0), 0);
}

BOLT_AUTO_TEST_CASE(TransformOnBigView) {
	int size = 200*200*200;
	DeviceImage<int, 2> result_image(1, size);
	auto checker_board = checkerboard(1, 0, Int2(2, 2), Int2(1, size));

	transform(checker_board, result_image.view(), IdentityFunctor());

	BOOST_CHECK_EQUAL(sumSquareDifferences(result_image.constView(), checker_board, 0), 0);
}

BOLT_AUTO_TEST_CASE(HostTransform) {
	HostImage<int, 2> result_image(16, 16);
	auto checker_board = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));

	transform(checker_board, result_image.view(), IdentityFunctor());

	BOOST_CHECK_EQUAL(sumSquareDifferences(result_image.constView(), checker_board, 0), 0);
}

#ifdef BOLT_USE_UNIFIED_MEMORY
BOLT_AUTO_TEST_CASE(UnifiedTransform) {
	UnifiedImage<int, 2> result_image(16, 16);
	auto checker_board = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));

	transform(checker_board, result_image.view(), IdentityFunctor());

	BOOST_CHECK_EQUAL(sumSquareDifferences(result_image.constView(), checker_board, 0), 0);
}
#endif


BOLT_AUTO_TEST_CASE(DeviceForEach) {
	DeviceImage<int, 2> device_image(16, 16);
	auto checker_board1 = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_board2 = checkerboard(2, 1, Int2(2, 2), Int2(16, 16));

	copy(checker_board1, device_image.view());

	forEach(device_image.view(), IncrementFunctor<int>(1));

	BOOST_CHECK_EQUAL(sumSquareDifferences(device_image.constView(), checker_board2, 0), 0);
}

BOLT_AUTO_TEST_CASE(ForeachOverBigView) {
	constexpr int size = 200*200*200;
	DeviceImage<int, 2> device_image(1, size);
	auto view = checkerboard(1, 0, Int2(1,1), Int2(1,size));
	auto resultingView = checkerboard(3, 2, Int2(1,1), Int2(1,size));
	copy(view, device_image.view());
	forEach(device_image.view(), IncrementFunctor<int>(2));
	auto testView = checkerboard(0, 0, Int2(1,1), Int2(1,1));
	BOOST_CHECK_EQUAL(sumSquareDifferences(device_image.view(), resultingView, 0), 0);
}


BOLT_AUTO_TEST_CASE(HostForEach) {
	HostImage<int, 2> host_image(16, 16);
	auto checker_board1 = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_board2 = checkerboard(2, 1, Int2(2, 2), Int2(16, 16));

	copy(checker_board1, host_image.view());

	forEach(host_image.view(), IncrementFunctor<int>(1));

	BOOST_CHECK_EQUAL(sumSquareDifferences(host_image.constView(), checker_board2, 0), 0);
}

#ifdef BOLT_USE_THREAD_BUILDING_BLOCKS
template <typename TView>struct ParallelForeachPolicy : public DefaultForEachPolicy<TView>
{
	static constexpr bool bParallelHost = true;
};
BOLT_AUTO_TEST_CASE(HostForEachParallel) {
	HostImage<int, 2> host_image(16, 16);
	auto checker_board1 = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_board2 = checkerboard(2, 1, Int2(2, 2), Int2(16, 16));

	copy(checker_board1, host_image.view());

	forEach(host_image.view(), IncrementFunctor<int>(1), ParallelForeachPolicy<decltype(host_image.view())>{});

	BOOST_CHECK_EQUAL(sumSquareDifferences(host_image.constView(), checker_board2, 0), 0);
}
#endif

#ifdef BOLT_USE_UNIFIED_MEMORY
BOLT_AUTO_TEST_CASE(UnifiedForEach) {
	UnifiedImage<int, 2> unified_image(16, 16);
	auto checker_board1 = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_board2 = checkerboard(2, 1, Int2(2, 2), Int2(16, 16));

	copy(checker_board1, unified_image.view());

	forEach(unified_image.view(), IncrementFunctor<int>(1));

	BOOST_CHECK_EQUAL(sumSquareDifferences(unified_image.constView(), checker_board2, 0), 0);
}
#endif

BOLT_AUTO_TEST_CASE(DeviceForEachPosition) {
	DeviceImage<int, 2> device_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	copy(source_view, device_image.view());
	auto size = source_view.size();

	forEachPosition(device_image.view(), [size]__device__(int& element, Int2 index){ element += getLinearAccessIndex(size, index);} );

	BOOST_CHECK_EQUAL(sumSquareDifferences(device_image.constView(), target_view, 0), 0);
}

BOLT_AUTO_TEST_CASE(ForeachPositionOverBigView) {
	constexpr int size = 200*200*200;
	DeviceImage<int, 2> device_image(1, size);
	auto source_view = makeConstantImageView(1, Int2(1, size));
	auto target_view = makeUniqueIdImageView( Int2(1, size), 1);
	copy(source_view, device_image.view());
	auto viewSize = source_view.size();
	forEachPosition(device_image.view(), [viewSize]__device__(int& element, const Int2& index){ element += getLinearAccessIndex(viewSize, index);});
	BOOST_CHECK_EQUAL(sumSquareDifferences(device_image.view(), target_view, 0), 0);
}


BOLT_AUTO_TEST_CASE(HostForEachPosition) {
	HostImage<int, 2> host_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	copy(source_view, host_image.view());
	auto size = source_view.size();
	forEachPosition(host_image.view(), [size](int& element, const Int2& index){ element += getLinearAccessIndex(size, index);});

	BOOST_CHECK_EQUAL(sumSquareDifferences(host_image.constView(), target_view, 0), 0);
}

#ifdef BOLT_USE_UNIFIED_MEMORY
BOLT_AUTO_TEST_CASE(UnifiedForEachPosition) {
	UnifiedImage<int, 2> unified_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	copy(source_view, unified_image.view());

	auto size = source_view.size();
	forEachPosition(
		unified_image.view(),
		[size]__device__(int& element, const Int2& index){
			element += getLinearAccessIndex(size, index);
		});

	BOOST_CHECK_EQUAL(sumSquareDifferences(unified_image.constView(), target_view, 0), 0);
}
#endif

#ifdef BOLT_USE_THREAD_BUILDING_BLOCKS
BOLT_AUTO_TEST_CASE(HostForEachPositionParallel) {
	HostImage<int, 2> host_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	copy(source_view, host_image.view());
	auto size = source_view.size();
	forEachPosition(host_image.view(), [size](int& element, const Int2& index){ element += getLinearAccessIndex(size, index);}, ParallelForeachPolicy<decltype(host_image.view())>());

	BOOST_CHECK_EQUAL(sumSquareDifferences(host_image.constView(), target_view, 0), 0);
}
#endif

BOLT_AUTO_TEST_CASE(DeviceTransformPosition) {

	DeviceImage<int, 2> result_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	auto size = source_view.size();
	transformPosition(source_view, result_image.view(), [size]__device__ __host__(int element, const Int2& index){return element + getLinearAccessIndex(size, index);});

	BOOST_CHECK_EQUAL(sumSquareDifferences(result_image.constView(), target_view, 0), 0);
}

BOLT_AUTO_TEST_CASE(TransformPositionOverBigView) {
	constexpr int size = 200*200*200;
	DeviceImage<int, 2> device_image(1, size);
	auto source_view = makeConstantImageView(1, Int2(1, size));
	auto target_view = makeUniqueIdImageView( Int2(1, size), 1);

	auto viewSize = source_view.size();
	transformPosition(source_view, device_image.view(), [viewSize]__device__ __host__(const int element, const Int2& index)->int{ return element + getLinearAccessIndex(viewSize, index);});
	BOOST_CHECK_EQUAL(sumSquareDifferences(device_image.view(), target_view, 0), 0);
}


BOLT_AUTO_TEST_CASE(HostTransformPosition) {
	HostImage<int, 2> host_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	auto size = source_view.size();
	transformPosition(source_view, host_image.view(), [size]__device__ __host__(const int element, const Int2& index)->int{ return element + getLinearAccessIndex(size, index);});

	BOOST_CHECK_EQUAL(sumSquareDifferences(host_image.constView(), target_view, 0), 0);
}

#ifdef BOLT_USE_UNIFIED_MEMORY
BOLT_AUTO_TEST_CASE(UnifiedTransformPosotion) {
	UnifiedImage<int, 2> unified_image(16, 16);
	auto source_view = makeConstantImageView(1, Int2(16, 16));
	auto target_view = makeUniqueIdImageView( Int2(16, 16), 1);

	auto size = source_view.size();
	transformPosition(
		source_view,
		unified_image.view(),
		[size]__device__ __host__(int element, const Int2& index)->int {
			return element + getLinearAccessIndex(size, index);
		});

	BOOST_CHECK_EQUAL(sumSquareDifferences(unified_image.constView(), target_view, 0), 0);
}
#endif

BOLT_AUTO_TEST_CASE(DimensionalReduce3D) {
	DeviceImage<int, 2> device_image(24, 16);
	auto input = makeConstantImageView(1, Int3(32, 24, 16));

	dimensionReduce(input, device_image.view(), DimensionValue<0>(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(
		sumSquareDifferences(device_image.constView(), makeConstantImageView(32, device_image.size()), 0),
		0);
}

BOLT_AUTO_TEST_CASE(DimensionalReduce3DOtherDimensions) {
	DeviceImage<int, 2> device_image(32, 16);
	auto input = makeConstantImageView(1, Int3(32, 24, 16));

	dimensionReduce(input, device_image.view(), DimensionValue<1>(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(
		sumSquareDifferences(device_image.constView(), makeConstantImageView(24, device_image.size()), 0),
		0);
}

BOLT_AUTO_TEST_CASE(DimensionalReduce3DLongRange) {
	DeviceImage<int, 2> device_image(4, 4);
	auto input = makeConstantImageView(1, Int3(4, 4, 2000));

	dimensionReduce(input, device_image.view(), DimensionValue<2>(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(
		sumSquareDifferences(device_image.constView(), makeConstantImageView(2000, device_image.size()), 0),
		0);
}

BOLT_AUTO_TEST_CASE(DimensionalReduce2D) {
	thrust::device_vector<int> device_vector(24);
	auto input = makeConstantImageView(1, Int2(32, 24));

	dimensionReduce(input, makeArrayView(device_vector), DimensionValue<0>(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(
		sumSquareDifferences(makeArrayConstView(device_vector), makeConstantImageView(32, device_vector.size()), 0),
		0);
}

}  // namespace bolt
