// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE ImageLocatorTest
#include <boost/test/included/unit_test.hpp>
#include <tests/test_utils.h>
#include <boltview/image_locator.h>
#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/subview.h>


namespace bolt {

BOLT_AUTO_TEST_CASE(BorderedView) {
	HostImage<int, 2> host_image(3, 3);
	host_image.clear();
	auto view = host_image.view();

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			view[Int2(i, j)] = i + 3 * j;
		}
	}
	auto original_locator = LocatorConstruction<BorderHandling::kRepeat>::create(host_image.constView(), Vector<int, 2> (1, 1));

	auto bordered_view = borderedSubview(host_image.constView(), Int2(1, 1), Int2(1,1));

	auto bordered_locator = LocatorConstruction<BorderHandling::kRepeat>::create(bordered_view, Int2());

	BOOST_CHECK_EQUAL(bordered_locator.get(), original_locator.get());
	BOOST_CHECK_EQUAL(bordered_locator[Int2(-1, -1)], original_locator[Int2(-1, -1)]);
}

BOLT_AUTO_TEST_CASE(BorderedViewFromViewBorderPolicy) {
	HostImage<int, 2> host_image(8, 8);
	host_image.clear();
	auto view = host_image.view();

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			view[Int2(i, j)] = i + 8 * j;
		}
	}
	auto original_locator = LocatorConstruction<BorderHandling::kMirror>::create(host_image.constView(), Int2());

	auto image_subview = subview(host_image.constView(), bolt::Int2(), bolt::Int2(4,8));

	auto bordered_view = borderedSubview(image_subview, Int2(2, 0), Int2(2, 8));

	auto bordered_locator = LocatorConstruction<BorderHandling::kMirror>::create(bordered_view, Int2());

	BOOST_CHECK_EQUAL(bordered_locator[Int2(2, 0)], original_locator[Int2(2, 0)]);
	BOOST_CHECK_EQUAL(bordered_locator[Int2(1, 0)], original_locator[Int2(3, 0)]);
	BOOST_CHECK_EQUAL(bordered_locator[Int2(-1, 0)], original_locator[Int2(1, 0)]);
}

}  // namespace bolt
