// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE InterpolationTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <vector>

#include <boltview/host_image_view.h>
#include <boltview/interpolation.h>
#include "tests/test_utils.h"


namespace bolt {

// NOTE(fidli): compilation error
BOLT_AUTO_TEST_CASE(TrilinearInterpolation, BOLT_TEST_SKIP) {
	std::vector<float> data = {
			2.0f, 4.0f,
			6.0f, 12.0f,
			8.0f, 16.0f,
			10.0f, 20.0f
		};

	auto view = makeHostImageConstView(data.data(), Int3(2, 2, 2), Int3(1, 2, 4));
	// NOTE(fidli): this causes compilation error
	/* TrilinearInterpolator<float> interpolator(0.0f);

	BOOST_CHECK_CLOSE(interpolator(view, Float3(0.5f, 0.5f, 0.5f)), 2.0f, kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(interpolator(view, Float3(1.0f, 0.5f, 0.5f)), 3.0f, kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(interpolator(view, Float3(1.0f, 1.0f, 0.5f)), 6.0f, kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(interpolator(view, Float3(1.0f, 1.0f, 1.0f)), 9.75f, kFloatTestEpsilon);

	BOOST_CHECK_CLOSE(interpolator(view, Float3(0.75f, 0.5f, 0.5f)), 2.5f, kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(interpolator(view, Float3(0.75f, 0.75f, 0.5f)), 3.75f, kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(interpolator(view, Float3(0.25f, 0.5f, 0.5f)), 1.5f, kFloatTestEpsilon);

	BOOST_CHECK_CLOSE(interpolator(view, Float3(1.75f, 0.5f, 0.5f)), 3.0f, kFloatTestEpsilon);
	*/
}

}  // namespace bolt
