// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE ImageStackTest
#include <boost/test/included/unit_test.hpp>

#include <boltview/image_stack.h>
#include <boltview/copy.h>
#include <boltview/for_each.h>
#include <boltview/transform.h>

#include "test_defs.h"
#include "test_utils.h"

namespace bolt {

struct TestLocatorFunctor {

	template<typename TLocator>
	BOLT_DECL_HYBRID
	int operator()(TLocator &loc) const {
		return 0;
	}

	template<typename TValue, typename TPosition>
	BOLT_DECL_HYBRID
	int operator()(TValue val, TPosition) const {
		return 0;
	}
};
#if 0
BOLT_AUTO_TEST_CASE(ImageStackBasicAlgorithms) {
	HostImage<int, 2> host_image(8, 8);
	ImageStackAdapter<HostImage<int, 3>> image_stack(Int3(8, 8, 4));

	ImageStackAdapter<DeviceImage<int, 3>> device_image_stack1(Int3(8, 8, 4));
	ImageStackAdapter<DeviceImage<int, 3>> device_image_stack2(Int3(8, 8, 4));
	// ImageStackAdapter<HostImage<int, 3>> device_image_stack1(Int3(8, 8, 4));
	// ImageStackAdapter<HostImage<int, 3>> device_image_stack2(Int3(8, 8, 4));

	ForEach(image_stack.view(), [](auto &value) { value = 42; });
	auto const_view = makeConstantImageView(42, Int3(8, 8, 4));

	BOOST_CHECK_EQUAL(SumSquareDifferences(image_stack.constView().Data(), const_view, 0), 0);

	copy(image_stack.constView(), device_image_stack1.view());

	TransformPosition(
		device_image_stack1.constView(),
		device_image_stack2.view(),
		TestLocatorFunctor{}/*[](auto locator) {
			return 0;
		}*/);
	TransformLocator(
		device_image_stack1.constView(),
		device_image_stack2.view(),
		TestLocatorFunctor{}/*[](auto locator) {
			return 0;
		}*/);
}
#endif //0
}  // namespace bolt
