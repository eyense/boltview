// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se


#define BOOST_TEST_MODULE ViewTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <algorithm>

#include <boltview/copy.h>
#include <boltview/create_view.h>
#include <boltview/cuda_utils.h>
#include <boltview/geometrical_transformation.h>
#include <boltview/host_image.h>
#include <boltview/host_image_view.h>
#include <boltview/device_image.h>
#include <boltview/device_image_view.h>
#include <boltview/reduce.h>
#include <boltview/subview.h>
#include <boltview/tests/test_utils.h>

namespace bolt {


BOLT_AUTO_TEST_CASE(CopyPaddedHostImage) {
  	DeviceImage<int, 3, LongIndexViewPolicy> device_image(3, 3, 3);
  	HostImage<int, 3, LongIndexViewPolicy> host_image(4, 4, 3);

  	auto host_view = host_image.view();
  	for (int64_t k = 0; k < 3; ++k) {
  	  	for (int64_t j = 0; j < 4; ++j) {
  	  	  	for (int64_t i = 0; i < 4; ++i) {
  	  	  	  	host_view[LongInt3(i, j, k)] = 1;
  	  	  	}
  	  	}
  	}

  	auto subimage_view = makeHostImageConstView(host_image.pointer(), LongInt3(3, 3, 3), host_image.strides(), LongIndexViewPolicy());
  	copy(subimage_view, device_image.view());
  	auto view = makeConstantImageView(1, LongInt3(3, 3, 3), LongIndexViewPolicy());
  	int diff = reduce(square(subtract(device_image.view(), view)), 0, thrust::plus<int>());
  	BOOST_CHECK_EQUAL(diff, 0);
}


BOLT_AUTO_TEST_CASE(FlatSum) {
	auto view1 = makeConstantImageView(3.0f, LongInt3(512, 512, 64), LongIndexViewPolicy());

	float sum = reduce(view1, 0.0f, thrust::plus<float>());
	BOOST_CHECK_CLOSE(sum, 3.0f * 512 * 512 * 64, kFloatTestEpsilon);
}

// To test values of procedural device view we need to copy it to the memory based device view
// and then to the host memory view, where we can access the values easily.
BOLT_AUTO_TEST_CASE(FlatCopyAndCopyToHost) {
	auto cview = makeConstantImageView(7.0f, LongInt3(16, 16, 2), LongIndexViewPolicy());
	DeviceImage<int, 3, LongIndexViewPolicy> device_image(16, 16, 2);
	HostImage<int, 3, LongIndexViewPolicy> host_image(16, 16, 2);

	copy(cview, view(device_image));

	float sum = reduce(constView(device_image), 0.0f, thrust::plus<float>());
	BOOST_CHECK_CLOSE(sum, 7.0f * cview.elementCount(), kFloatTestEpsilon);

	copy(constView(device_image), view(host_image));

	auto host_view = constView(host_image);
	for (int i = 0; i < host_view.elementCount(); ++i) {
		BOOST_CHECK_CLOSE(7.0f, linearAccess(host_view, i), kFloatTestEpsilon);
	}
}


BOLT_AUTO_TEST_CASE(CheckerBoardSum) {
	auto checker_view = checkerboard(1, 0, LongInt2(2, 2), LongInt2(16, 16), LongIndexViewPolicy());

	float sum = reduce(checker_view, 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 16 * 16 / 2);
}


BOLT_AUTO_TEST_CASE(CheckerBoardAdditionSum) {
	auto checker_view = checkerboard(
		1, 0, LongInt2(2, 2), LongInt2(16, 16), LongIndexViewPolicy());
	auto checker_view2 = checkerboard(
  	  	4, 0, LongInt2(4, 4), LongInt2(16, 16), LongIndexViewPolicy());

	float sum = reduce(add(checker_view, checker_view2), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 5 * 16 * 16 / 2);
}


BOLT_AUTO_TEST_CASE(CopiedCheckerBoardSum) {
	auto checker_view = checkerboard(
  	  	int(1), 0, LongInt2(2, 2), LongInt2(16, 16), LongIndexViewPolicy());
	DeviceImage<int, 2, LongIndexViewPolicy> image(16, 16);
	static const int kExpectedSum = 16 * 16 / 2;
	int sum1 = reduce(checker_view, 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(kExpectedSum, sum1);

	copy(checker_view, image.view());
	int sum2 = reduce(image.constView(), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(kExpectedSum, sum2);
}


BOLT_AUTO_TEST_CASE(MultipliedCheckerBoardSum) {
	auto view1 = checkerboard(
  	  	1, 0, LongInt2(2, 2), LongInt2(16, 16), LongIndexViewPolicy());
	auto view2 = checkerboard(
  	  	2, 1, LongInt2(8, 8), LongInt2(16, 16), LongIndexViewPolicy());

	float sum = reduce(multiply(view1, view2), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 16 * 16 / 2 + 16 * 16 / 4);
}


BOLT_AUTO_TEST_CASE(Multiplication) {
	// Multiply constant image by checkerboard -> result should be again checkerboard, but with multiplied values
	auto view1 = makeConstantImageView(5, LongInt2(16, 16), LongIndexViewPolicy());
	auto view2 = checkerboard(
  	  	4, 0, LongInt2(8, 8), LongInt2(16, 16), LongIndexViewPolicy());
	auto multiplication = multiply(view1, view2);
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(view1.size());
	HostImage<int, 2, LongIndexViewPolicy> host_image(view1.size());

	copy(multiplication, device_image.view());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (int64_t j = 0; j < host_view.size()[1]; ++j) {
		for (int64_t i = 0; i < host_view.size()[0]; ++i) {
			LongInt2 index(i, j);
			// compute value according to the checkerboard pattern
			int result = (sum(div(index, Int2(8, 8))) % 2) ? 20 : 0;
			BOOST_CHECK_EQUAL(result, host_view[index]);
		}
	}
}


BOLT_AUTO_TEST_CASE(SquareTest) {
	auto cview = makeConstantImageView(5, LongInt2(16, 16), LongIndexViewPolicy());
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(cview.size());
	HostImage<int, 2, LongIndexViewPolicy> host_image(cview.size());

	copy(square(cview), device_image.view());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (int64_t j = 0; j < host_view.size()[1]; ++j) {
		for (int64_t i = 0; i < host_view.size()[0]; ++i) {
			LongInt2 index(i, j);
			BOOST_CHECK_EQUAL(25, host_view[index]);
		}
	}
}


BOLT_AUTO_TEST_CASE(MirrorTest) {
	auto flat = makeConstantImageView(4, LongInt2(16, 16), LongIndexViewPolicy());
	auto view1 = checkerboard(
  	  	4, 0, LongInt2(2, 2), LongInt2(16, 16), LongIndexViewPolicy());
	auto view2 = mirror(view1, Bool2(true, false));

	// Addition of checkerboard and its mirror view should fill the plane - hence the subtraction of the constant image
	int square_difference = reduce(square(subtract(flat, add(view1, view2))), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(0, square_difference);
}


BOLT_AUTO_TEST_CASE(MirrorFromHostData) {
	HostImage<int, 2, LongIndexViewPolicy> host_image(4, 4);
	auto host_view = host_image.view();

	for (int64_t j = 0; j < 4; ++j) {
		for (int64_t i = 0; i < 4; ++i) {
			host_view[LongInt2(i, j)] = i + j;
		}
	}
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(4, 4);
	copy(host_view, device_image.view());

	DeviceImage<int, 2, LongIndexViewPolicy> mirrored_image(4, 4);
	copy(mirror(device_image.constView(), Bool2(true, true)), mirrored_image.view());

	copy(mirrored_image.constView(), host_view);
	for (int64_t j = 0; j < 4; ++j) {
		for (int64_t i = 0; i < 4; ++i) {
			BOOST_CHECK_EQUAL(host_view[LongInt2(4 - i - 1, 4 - j - 1)], i + j);
		}
	}
}


BOLT_AUTO_TEST_CASE(Padding) {
	auto flat = makeConstantImageView(13, LongInt2(8, 8), LongIndexViewPolicy());
	// Combine two different paddings to create checkerboard then compare with actual checkerboard
	auto padded_view1 = paddedView(flat, LongInt2(16, 16), LongInt2(0, 0), 0);
	auto padded_view2 = paddedView(flat, LongInt2(16, 16), LongInt2(8, 8), 0);
	auto combined = add(padded_view1, padded_view2);
	auto board = checkerboard(
  	  	0, 13, LongInt2(8, 8), LongInt2(16, 16), LongIndexViewPolicy());

	int square_difference = reduce(square(subtract(board, combined)), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(0, square_difference);
}


BOLT_AUTO_TEST_CASE(PaddingFromHostData) {
	HostImage<int, 2, LongIndexViewPolicy> host_image(4, 4);
	auto host_view = host_image.view();

	for (int64_t j = 0; j < 4; ++j) {
		for (int64_t i = 0; i < 4; ++i) {
			host_view[LongInt2(i, j)] = 7;
		}
	}
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(4, 4);
	copy(host_view, device_image.view());

	DeviceImage<int, 2, LongIndexViewPolicy> padded_image(8, 8);
	copy(paddedView(device_image.constView(), padded_image.size(), LongInt2(2, 2), -1), padded_image.view());

	HostImage<int, 2, LongIndexViewPolicy> result_image(8, 8);
	auto result_view = result_image.view();

	copy(padded_image.constView(), result_view);
	for (int64_t j = 0; j < 8; ++j) {
		for (int64_t i = 0; i < 8; ++i) {
			bool out = i < 2 || j < 2 || i > 5 || j > 5;
			BOOST_CHECK_EQUAL(result_view[LongInt2(i, j)], out ? -1 : 7);
		}
	}
}


BOLT_AUTO_TEST_CASE(RotatedView) {
	auto pattern = checkerboard<float, 3, LongIndexViewPolicy>(
  	  	1.0f, 13.0f, LongInt3(2, 2, 2), LongInt3(3, 3, 3));
	auto padded_view = paddedView(mirror(pattern, Bool3(false, true, false)), LongInt3(6, 6, 6), LongInt3(), 0.0f);
	auto rotated_view = rotatedView(pattern, rotationQuaternion(float(-kPi / 2), Float3(0.0f, 0.0f , 1.0f)), 0.5f * pattern.size(), LongInt3(6, 6, 6));

	testViewsForIdentity(padded_view, rotated_view);
}


BOLT_AUTO_TEST_CASE(MeshGridTest) {
	auto grid = meshGrid(LongInt3(), LongInt3(4, 4, 4), LongIndexViewPolicy());
	DeviceImage<int, 3, LongIndexViewPolicy> device_image(grid[0].size());
	HostImage<int, 3, LongIndexViewPolicy> host_image(grid[0].size());
	auto host_view = host_image.constView();
	for (int dim = 0; dim < 3; ++dim) {
		copy(grid[dim], device_image.view());
		copy(device_image.constView(), host_image.view());
		for (int64_t k = 0; k < 4; ++k) {
			for (int64_t j = 0; j < 4; ++j) {
				for (int64_t i = 0; i < 4; ++i) {
					LongInt3 coords(i, j, k);
					BOOST_CHECK_EQUAL(host_view[coords], coords[dim]);
				}
			}
		}
	}
}

// Subview of hybrid image is not supported
BOLT_AUTO_TEST_CASE(SubviewOnDevice, BOLT_TEST_SKIP) {
	auto meshgrid = meshGrid(LongInt3(), LongInt3(8, 8, 8), LongIndexViewPolicy());
	auto small_meshgrid = meshGrid(LongInt3(2, 2, 2), LongInt3(6, 6, 6), LongIndexViewPolicy());

	for (int64_t i = 0; i < 3; ++i) {
		auto mesh_subview = subview(meshgrid[i], LongInt3(2, 2, 2), LongInt3(4, 4, 4));
		DeviceImage<int, 3, LongIndexViewPolicy> device_subimage(4, 4, 4);
		copy(mesh_subview, device_subimage.view());

		BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_subview, small_meshgrid[i]), 0);
		BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_subimage.constView(), small_meshgrid[i]), 0);
	}
}

// Subview of hybrid image is not supported
BOLT_AUTO_TEST_CASE(CopySubviewOnDevice, BOLT_TEST_SKIP) {
	auto pattern = checkerboard(0, 1, LongInt2(4, 4), LongInt2(8, 8), LongIndexViewPolicy());
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(8, 8);
	copy(pattern, device_image.view());

	auto subpattern = subview(pattern, LongInt2(2, 2), LongInt2(4, 4));
	auto small_pattern = checkerboard(0, 1, LongInt2(2, 2), LongInt2(4, 4), LongIndexViewPolicy());

	DeviceImage<int, 2, LongIndexViewPolicy> device_subimage(4, 4);
	copy(subview(device_image.constView(), LongInt2(2, 2), LongInt2(4, 4)), device_subimage.view());
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(small_pattern, device_subimage.constView()), 0);
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(small_pattern, subpattern), 0);
}


BOLT_AUTO_TEST_CASE(CopySubviewToHost) {
	auto pattern = checkerboard(1, 2, LongInt2(4, 4), LongInt2(8, 8), LongIndexViewPolicy());
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(8, 8);
	copy(pattern, device_image.view());


	HostImage<int, 2, LongIndexViewPolicy> host_image(8, 8);
	host_image.clear();

	copy(
		subview(device_image.constView(), LongInt2(2, 2), LongInt2(4, 4)),
		subview(host_image.constView(), LongInt2(4, 4), LongInt2(4, 4)));

	auto host_view = host_image.constView();
	for (int64_t j = 0; j < 8; ++j) {
		for (int64_t i = 0; i < 8; ++i) {
			if (i < 4 || j < 4) {
				BOOST_CHECK_EQUAL(host_view[LongInt2(i, j)], 0);
			} else {
				BOOST_CHECK_EQUAL(host_view[LongInt2(i, j)], sum(div(Int2(i, j), Int2(2, 2))) % 2 ? 1 : 2);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(CopySubviewToHost3D) {
	DeviceImage<int, 3, LongIndexViewPolicy> device_image(32, 32, 32);
	copy(makeConstantImageView(int(5), device_image.size(), LongIndexViewPolicy()), device_image.view());

	HostImage<int, 3, LongIndexViewPolicy> host_image(32, 32, 32);
	host_image.clear();
	copy(
		subview(device_image.constView(), LongInt3(), LongInt3(9, 17, 4)),
		subview(host_image.constView(), LongInt3(), LongInt3(9, 17, 4)));

	auto host_view = host_image.constView();
	for (int64_t k = 0; k < host_view.size()[2]; ++k) {
		for (int64_t j = 0; j < host_view.size()[1]; ++j) {
			for (int64_t i = 0; i < host_view.size()[0]; ++i) {
				if (i < 9 && j < 17 && k < 4) {
					BOOST_CHECK_EQUAL(host_view[LongInt3(i, j, k)], 5);
				} else {
					BOOST_CHECK_EQUAL(host_view[LongInt3(i, j, k)], 0);
				}
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(CastView) {
  	HostImage<int, 2, LongIndexViewPolicy> host_image(4, 4);
	auto host_view = host_image.view();

	for (int64_t j = 0; j < 4; ++j) {
		for (int64_t i = 0; i < 4; ++i) {
			host_view[LongInt2(i, j)] = i + j;
		}
	}

  	auto char_view = cast<char>(host_view);

	for (int64_t j = 0; j < 4; ++j) {
		for (int64_t i = 0; i < 4; ++i) {
			BOOST_CHECK_EQUAL(char_view[LongInt2(i, j)], i + j);
		}
	}
}

BOLT_AUTO_TEST_CASE(CopySliceOnDevice) {
	auto pattern = checkerboard(0, 1, LongInt3(4, 4, 4), LongInt3(8, 8, 8), LongIndexViewPolicy());
	DeviceImage<int, 3, LongIndexViewPolicy> device_image(8, 8, 8);
	copy(pattern, device_image.view());

	auto pattern_slice = slice<1>(pattern, 1);
	auto pattern_2d = checkerboard(0, 1, LongInt2(4, 4), LongInt2(8, 8), LongIndexViewPolicy());

	DeviceImage<int, 2, LongIndexViewPolicy> device_slice_image(8, 8);
	copy(slice<1>(device_image.constView(), 1), device_slice_image.view());

	BOOST_CHECK_EQUAL(squareDifferenceOfImages(pattern_2d, device_slice_image.constView()), 0);
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(pattern_2d, pattern_slice), 0);
}


BOLT_AUTO_TEST_CASE(SliceOnDevice) {
	// meshgrid is used to generate 3 axis oriented image gradients
	// Slice in all 3 directions and check if we get correct gradient
	// or constant image in case of cut perpendicular to the gradient direction.
	auto meshgrid = meshGrid(LongInt3(), LongInt3(8, 8, 8), LongIndexViewPolicy());
	auto slice_meshgrid = meshGrid(LongInt2(), LongInt2(8, 8), LongIndexViewPolicy());

	for (int64_t i = 0; i < 3; ++i) {
		auto mesh_slice = slice<1>(meshgrid[i], 3);
		DeviceImage<int, 2, LongIndexViewPolicy> device_slice_image(8, 8);
		copy(mesh_slice, device_slice_image.view());

		if (i != 1) {
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_slice, slice_meshgrid[std::min(i, int64_t(1))]), 0);
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_slice_image.constView(), slice_meshgrid[std::min(i, int64_t(1))]), 0);
		} else {
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_slice, makeConstantImageView(3, mesh_slice.size(), LongIndexViewPolicy())), 0);
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_slice_image.constView(), makeConstantImageView(3, mesh_slice.size(), LongIndexViewPolicy())), 0);
		}
	}
}

BOLT_AUTO_TEST_CASE(ViewSizeException) {
	LongInt2 size2_a, size2_b(1, 2);
	bool did_throw = false;
	try {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(size2_a, size2_b));
	} catch (boost::exception &e) {
		// std::string const *extra  = boost::get_error_info<tag_index_view_pair_sizes>(e);
		std::pair<LongInt2, LongInt2> const *extra = boost::get_error_info<ViewPairSizesErrorInfoLongInt2>(e);
		BOOST_CHECK_EQUAL(extra->first, size2_a);
		BOOST_CHECK_EQUAL(extra->second, size2_b);
		did_throw = true;
	}
	BOOST_CHECK(did_throw);
}

BOLT_AUTO_TEST_CASE(MinMaxTest) {
	const LongInt2 testDimensions(16, 16);
	std::array<int, 16 * 16> testData;
	testData.fill(5);
	auto view = makeHostImageView(testData.data(), testDimensions, LongIndexViewPolicy());
	view[LongInt2(4, 4)] = 1;
	view[LongInt2(15, 0)] = 10;
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(view.size());
	copy(view, device_image.view());
	Int2 min_max = minMax(device_image.constView());
	BOOST_CHECK_EQUAL(1, min_max[0]);
	BOOST_CHECK_EQUAL(10, min_max[1]);
}

BOLT_AUTO_TEST_CASE(MinMaxMeanTest) {
	const LongInt2 testDimensions(16, 16);
	std::array<int, 16 * 16> testData;
	testData.fill(5);
	auto view = makeHostImageView(testData.data(), testDimensions, LongIndexViewPolicy());
	view[LongInt2(4, 4)] = 1;
	view[LongInt2(15, 0)] = 10;
	DeviceImage<int, 2, LongIndexViewPolicy> device_image(view.size());
	copy(view, device_image.view());
	auto min_max_mean = minMaxMean(device_image.constView());
	BOOST_CHECK_EQUAL(1, min_max_mean.template get<0>());
	BOOST_CHECK_EQUAL(10, min_max_mean.template get<1>());
	BOOST_CHECK_EQUAL(5.00390625, min_max_mean.template get<2>());
}

BOLT_AUTO_TEST_CASE(IntOverflowTest) {
  	const int64_t tile_width = int64_t(1) << 31;
  	auto view = checkerboard(
  	  	1, 0, LongInt2(tile_width, 1), LongInt2(tile_width * 8, 4), LongIndexViewPolicy());
  	for (int64_t i = tile_width - 2; i < tile_width + 2; i++) {
  	  	for (int64_t j = 0; j < 4; j++) {
  	  	  	BOOST_CHECK_EQUAL(view[LongInt2(i, j)], (i / tile_width + j) % 2);
  	  	}
  	}

  	for (int64_t i = 4 * tile_width - 2; i < 4 * tile_width + 2; i++) {
  	  	for (int64_t j = 0; j < 4; j++) {
  	  	  	BOOST_CHECK_EQUAL(view[LongInt2(i, j)], (i / tile_width + j) % 2);
  	  	}
  	}
}


}  // namespace bolt
