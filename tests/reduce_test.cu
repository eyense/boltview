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

}  // namespace bolt
