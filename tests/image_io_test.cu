// Copyright 2017 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#define BOOST_TEST_MODULE ImageIOTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <boltview/image_io.h>
#include <boltview/host_image.h>
#include <boltview/math/vector.h>
#include <boltview/subview.h>

#include "texture_image_test_utils.h"
#include "test_utils.h"

namespace bolt {

template<typename TImage>
void testDumpAndLoadImage(
	typename TImage::SizeType size,
	std::string prefix)
{
	TImage input(size);
	TImage output(size);
	generateRandomView(input.view(), 1337ULL);

	dump(input.view(), prefix);
	load(output.view(), prefix);
	testViewsForIdentity(input.view(), output.view());

	auto result = load<TImage>(size, prefix);
	testViewsForIdentity(input.view(), result.view());
}

template<typename TImage>
void testDumpAndLoadSubview(
	typename TImage::SizeType size,
	typename TImage::SizeType corner,
	typename TImage::SizeType subview_size,
	std::string prefix)
{
	TImage input(size);
	TImage output(size);
	generateRandomView(input.view(), 1337ULL);
	auto input_subview = subview(input.view(), corner, subview_size);
	auto output_subview = subview(output.view(), corner, subview_size);

	dump(input_subview, prefix);
	load(output_subview, prefix);
	testViewsForIdentity(input_subview, output_subview);

	auto result = load<TImage>(subview_size, prefix);
	testViewsForIdentity(input_subview, result.view());
}

BOLT_AUTO_TEST_CASE(DumpAndLoadImage) {
	testDumpAndLoadImage<HostImage<float, 2>>(Int2(12, 16), "test");
}

BOLT_AUTO_TEST_CASE(DumpAndLoadSubview) {
	Int3 size(12, 16, 18);
	Int3 corner(6, 0, 0);
	Int3 subview_size(6, 16, 18);
	testDumpAndLoadSubview<HostImage<float, 3>>(size, corner, subview_size, "test_subview");
}

}  // namespace bolt
