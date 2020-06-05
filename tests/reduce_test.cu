// Copyright 2020 Eyen SE
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

BOLT_AUTO_TEST_CASE(ConstantViewReduce) {

	auto cview_2d = makeConstantImageView(1, Int2(800, 800));
	auto cview_3d = makeConstantImageView(1, Int3(800, 800, 200));


	auto val1 = reduce(cview_2d, 0, thrust::plus<int>());
	auto val2 = reduce(cview_3d, 0, thrust::plus<int>());


	BOOST_CHECK_EQUAL(val1, product(cview_2d.size()));
	BOOST_CHECK_EQUAL(val2, product(cview_3d.size()));
}

struct PlusInt2 {
	BOLT_DECL_HYBRID
	Int2 operator()(const Int2& x, const Int2& y) const {
		return x + y;
	}
};

BOLT_AUTO_TEST_CASE(ConstantViewReduceComplexType) {

	auto cview_2d = makeConstantImageView(Int2(1, -2), Int2(800, 800));
	auto cview_3d = makeConstantImageView(Int2(1, -2), Int3(800, 800, 200));


	auto val1 = reduce(cview_2d, Int2(0, 0), PlusInt2());
	auto val2 = reduce(cview_3d, Int2(0, 0), PlusInt2());


	BOOST_CHECK_EQUAL(val1[0], product(cview_2d.size()));
	BOOST_CHECK_EQUAL(val1[1], -2 * product(cview_2d.size()));
	BOOST_CHECK_EQUAL(val2[0], product(cview_3d.size()));
	BOOST_CHECK_EQUAL(val2[1], -2 * product(cview_3d.size()));
}


BOLT_AUTO_TEST_CASE(ConstantViewDimensionReduce) {

	auto cview_2d = makeConstantImageView(1, Int2(800, 800));
	auto cview_3d = makeConstantImageView(1, Int3(800, 800, 200));

	DeviceImage<int, 1> results_2d (800);
	DeviceImage<int, 2> results_3d (800, 200);

	HostImage<int, 1> host_results_2d (800);
	HostImage<int, 2> host_results_3d (800, 200);

	dimensionReduce(cview_2d, view(results_2d), DimensionValue<0>(), 0, thrust::plus<int>());
	dimensionReduce(cview_3d, view(results_3d), DimensionValue<1>(), 0, thrust::plus<int>());

	copy(constView(results_2d), view(host_results_2d));
	copy(constView(results_3d), view(host_results_3d));

	auto host_view_2d = constView(host_results_2d);
	auto host_view_3d = constView(host_results_3d);

	for (int x = 0; x < 800; ++x) {
		BOOST_CHECK_EQUAL(host_view_2d[x], 800);
	}

	for (int y = 0; y < 200; ++y) {
		for (int x = 0; x < 800; ++x) {
			BOOST_CHECK_EQUAL((host_view_3d[{x, y}]), 800);
		}
	}
}

}  // namespace bolt
