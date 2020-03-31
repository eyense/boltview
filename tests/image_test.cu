// Copyright 2017 Eyen SE
// Author: Lukas Marsalek lukas.marsalek@eyen.eu

#define BOOST_TEST_MODULE ImageTest
#include <boost/test/included/unit_test.hpp>
#include "tests/test_utils.h"

#include <cuda.h>

#include <boltview/create_view.h>
#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/reduce.h>
#include <boltview/copy.h>
#include <boltview/detail/meta_algorithm_utils.h>
#include <boltview/view_policy.h>

namespace bolt {

template<typename TElement = int>
std::pair<TElement, TElement> testPatternDimensions(){
	return std::pair<TElement, TElement>(32, 32);
}

template<typename TElement = int>
int testPatternReductionValue(){
	std::pair<TElement, TElement> dimensions = testPatternDimensions<TElement>();
	return int(std::get<0>(dimensions) * std::get<1>(dimensions)) * 2;
}

template<class TView>
void testPattern(TView buffer){
	using TIndex = typename TView::TIndex;
	for(TIndex j = 0; j < buffer.size()[1]; ++j){
		for(TIndex i = 0; i < buffer.size()[0]; ++i){
			buffer[typename TView::IndexType(i, j)] = 2;
		}
	}
}

BOLT_AUTO_TEST_CASE(ContainerStorage){
	const int numTestImages = 3;
	std::vector<DeviceImage<int, 2>> container;
	HostImage<int, 2> hostTestPattern(
			std::get<0>(testPatternDimensions()),
			std::get<1>(testPatternDimensions())
	);
	testPattern(hostTestPattern.view());
	container.resize(numTestImages);
	for(int i = 0; i < numTestImages; ++i){
		container[i] = DeviceImage<int, 2>(
			std::get<0>(testPatternDimensions()),
			std::get<1>(testPatternDimensions())
		);
		copy(hostTestPattern.constView(), container[i].view());
	};
	for(int i = 0; i < container.size(); ++i){
		int reduction = reduce(container[i].constView(), 0, thrust::plus<int>());
		BOOST_CHECK_EQUAL(reduction, testPatternReductionValue());
	}
}

BOLT_AUTO_TEST_CASE(DeviceImageMove) {
	{
		DeviceImage<float, 3> image(10, 10, 10);
		auto pointer = image.pointer();

		DeviceImage<float, 3> moved_to(std::move(image));

		BOOST_CHECK_EQUAL(image.pointer(), static_cast<decltype(pointer)>(nullptr));
		BOOST_CHECK_EQUAL(moved_to.pointer(), pointer);
	}

	{
		DeviceImage<float, 3> image(10, 10, 10);
		auto pointer = image.pointer();

		DeviceImage<float, 3> moved_to(6, 6, 6);
		moved_to = std::move(image);

		BOOST_CHECK_EQUAL(image.pointer(), static_cast<decltype(pointer)>(nullptr));
		BOOST_CHECK_EQUAL(moved_to.pointer(), pointer);
	}
}

BOLT_AUTO_TEST_CASE(HostImageMove) {
	{
		HostImage<float, 3> image(10, 10, 10);
		auto pointer = image.pointer();

		HostImage<float, 3> moved_to(std::move(image));

		BOOST_CHECK_EQUAL(image.pointer(), static_cast<decltype(pointer)>(nullptr));
		BOOST_CHECK_EQUAL(moved_to.pointer(), pointer);
	}

	{
		HostImage<float, 3> image(10, 10, 10);
		auto pointer = image.pointer();

		HostImage<float, 3> moved_to(6, 6, 6);
		moved_to = std::move(image);

		BOOST_CHECK_EQUAL(image.pointer(), static_cast<decltype(pointer)>(nullptr));
		BOOST_CHECK_EQUAL(moved_to.pointer(), pointer);
	}
}


BOLT_AUTO_TEST_CASE(ContainerStorageLongIndices){
	const int numTestImages = 3;
	std::vector<DeviceImage<int, 2, LongIndexViewPolicy>> container;
	HostImage<int, 2, LongIndexViewPolicy> hostTestPattern(
			std::get<0>(testPatternDimensions<int64_t>()),
			std::get<1>(testPatternDimensions<int64_t>())
	);
	testPattern(hostTestPattern.view());
	container.resize(numTestImages);
	for(int i = 0; i < numTestImages; ++i){
		container[i] = DeviceImage<int, 2, LongIndexViewPolicy>(
			std::get<0>(testPatternDimensions<int64_t>()),
			std::get<1>(testPatternDimensions<int64_t>())
		);
		// BOOST_CHECK_EQUAL(container[i].view().size(), hostTestPattern.constView().size());
		copy(constView(hostTestPattern), container[i].view());
	};

	for(int i = 0; i < container.size(); ++i){
		int reduction = reduce(container[i].constView(), 0, thrust::plus<int>());
		BOOST_CHECK_EQUAL(reduction, testPatternReductionValue());
	}
}


}  // namespace bolt
