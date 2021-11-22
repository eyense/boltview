// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se


#define BOOST_TEST_MODULE ViewTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>


#include <algorithm>

#include <boltview/create_view.h>
#include <boltview/cuda_utils.h>
#include <boltview/device_image.h>
#include <boltview/device_image_view.h>
#include <boltview/host_image.h>
#include <boltview/host_image_view.h>
#include <boltview/reduce.h>
#include <boltview/copy.h>
#include <boltview/geometrical_transformation.h>
#include <boltview/subview.h>

#include <boltview/tests/test_utils.h>

namespace bolt {

BOLT_AUTO_TEST_CASE(CopyPaddedHostImage) {
	DeviceImage<int, 3> device_image(3, 3, 3);
	HostImage<int, 3> host_image(4, 4, 3);

	auto host_view = host_image.view();
	for (int k = 0; k < 3; ++k) {
		for (int j = 0; j < 4; ++j) {
			for (int i = 0; i < 4; ++i) {
				host_view[Int3(i, j, k)] = 1;
			}
		}
	}

	auto subimage_view = makeHostImageConstView(host_image.pointer(), Int3(3, 3, 3), host_image.strides());
	copy(subimage_view, device_image.view());
	auto cview = makeConstantImageView(1, Int3(3, 3, 3));
	int diff = reduce(square(subtract(device_image.view(), cview)), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(diff, 0);
}


BOLT_AUTO_TEST_CASE(FlatSum) {
	auto view1 = makeConstantImageView(3.0f, Int3(512, 512, 64));

	float sum = reduce(view1, 0.0f, thrust::plus<float>());
	BOOST_CHECK_CLOSE(sum, 3.0f * 512 * 512 * 64, kFloatTestEpsilon);
}

// To test values of procedural device view we need to copy it to the memory based device view
// and then to the host memory view, where we can access the values easily.
BOLT_AUTO_TEST_CASE(FlatCopyAndCopyToHost) {
	auto cview = makeConstantImageView(7.0f, Int3(16, 16, 2));
	DeviceImage<int, 3> device_image(16, 16, 2);
	HostImage<int, 3> host_image(16, 16, 2);

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
	auto checker_view = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));

	float sum = reduce(checker_view, 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 16 * 16 / 2);
}


BOLT_AUTO_TEST_CASE(CheckerBoardAdditionSum) {
	auto checker_view = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_view2 = checkerboard(4, 0, Int2(4, 4), Int2(16, 16));

	float sum = reduce(add(checker_view, checker_view2), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 5 * 16 * 16 / 2);
}


BOLT_AUTO_TEST_CASE(CopiedCheckerBoardSum) {
	auto checker_view = checkerboard(int(1), 0, Int2(2, 2), Int2(16, 16));
	DeviceImage<int, 2> image(16, 16);
	static const int kExpectedSum = 16 * 16 / 2;
	int sum1 = reduce(checker_view, 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(kExpectedSum, sum1);

	copy(checker_view, view(image));
	int sum2 = reduce(constView(image), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(kExpectedSum, sum2);
}


BOLT_AUTO_TEST_CASE(MultipliedCheckerBoardSum) {
	auto checker_view1 = checkerboard(1, 0, Int2(2, 2), Int2(16, 16));
	auto checker_view2 = checkerboard(2, 1, Int2(8, 8), Int2(16, 16));

	float sum = reduce(multiply(checker_view1, checker_view2), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(sum, 16 * 16 / 2 + 16 * 16 / 4);
}


BOLT_AUTO_TEST_CASE(Multiplication) {
	// Multiply constant image by checkerboard -> result should be again checkerboard, but with multiplied values
	auto checker_view1 = makeConstantImageView(5, Int2(16, 16));
	auto checker_view2 = checkerboard(4, 0, Int2(8, 8), Int2(16, 16));
	auto multiplication = multiply(checker_view1, checker_view2);
	DeviceImage<int, 2> device_image(checker_view1.size());
	HostImage<int, 2> host_image(checker_view1.size());

	copy(multiplication, device_image.view());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (int j = 0; j < host_view.size()[1]; ++j) {
		for (int i = 0; i < host_view.size()[0]; ++i) {
			Int2 index(i, j);
			// compute value according to the checkerboard pattern
			int result = (sum(div(index, Int2(8, 8))) % 2) ? 20 : 0;
			BOOST_CHECK_EQUAL(result, host_view[index]);
		}
	}
}


BOLT_AUTO_TEST_CASE(SquareTest) {
	auto cview = makeConstantImageView(5, Int2(16, 16));
	DeviceImage<int, 2> device_image(cview.size());
	HostImage<int, 2> host_image(cview.size());

	copy(square(cview), device_image.view());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (int j = 0; j < host_view.size()[1]; ++j) {
		for (int i = 0; i < host_view.size()[0]; ++i) {
			Int2 index(i, j);
			BOOST_CHECK_EQUAL(25, host_view[index]);
		}
	}
}


BOLT_AUTO_TEST_CASE(MirrorTest) {
	auto flat = makeConstantImageView(4, Int2(16, 16));
	auto checker_view1 = checkerboard(4, 0, Int2(2, 2), Int2(16, 16));
	auto checker_view2 = mirror(checker_view1, Bool2(true, false));

	// Addition of checkerboard and its mirror view should fill the plane - hence the subtraction of the constant image
	int square_difference = reduce(square(subtract(flat, add(checker_view1, checker_view2))), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(0, square_difference);
}


BOLT_AUTO_TEST_CASE(MirrorFromHostData) {
	HostImage<int, 2> host_image(4, 4);
	auto host_view = host_image.view();

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			host_view[Int2(i, j)] = i + j;
		}
	}
	DeviceImage<int, 2> device_image(4, 4);
	copy(host_view, device_image.view());

	DeviceImage<int, 2> mirrored_image(4, 4);
	copy(mirror(device_image.constView(), Bool2(true, true)), mirrored_image.view());

	copy(mirrored_image.constView(), host_view);
	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			BOOST_CHECK_EQUAL(host_view[Int2(4 - i - 1, 4 - j - 1)], i + j);
		}
	}
}


BOLT_AUTO_TEST_CASE(Padding) {
	auto flat = makeConstantImageView(13, Int2(8, 8));
	// Combine two different paddings to create checkerboard then compare with actual checkerboard
	auto padded_view1 = paddedView(flat, Int2(16, 16), Int2(0, 0), 0);
	auto padded_view2 = paddedView(flat, Int2(16, 16), Int2(8, 8), 0);
	auto combined = add(padded_view1, padded_view2);
	auto board = checkerboard(0, 13, Int2(8, 8), Int2(16, 16));

	int square_difference = reduce(square(subtract(board, combined)), 0, thrust::plus<int>());
	BOOST_CHECK_EQUAL(0, square_difference);
}


BOLT_AUTO_TEST_CASE(PaddingFromHostData) {
	HostImage<int, 2> host_image(4, 4);
	auto host_view = host_image.view();

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			host_view[Int2(i, j)] = 7;
		}
	}
	DeviceImage<int, 2> device_image(4, 4);
	copy(host_view, device_image.view());

	DeviceImage<int, 2> padded_image(8, 8);
	copy(paddedView(device_image.constView(), padded_image.size(), Int2(2, 2), -1), padded_image.view());

	HostImage<int, 2> result_image(8, 8);
	auto result_view = result_image.view();

	copy(padded_image.constView(), result_view);
	for (int j = 0; j < 8; ++j) {
		for (int i = 0; i < 8; ++i) {
			bool out = i < 2 || j < 2 || i > 5 || j > 5;
			BOOST_CHECK_EQUAL(result_view[Int2(i, j)], out ? -1 : 7);
		}
	}
}

BOLT_AUTO_TEST_CASE(RotatedView) {
	auto pattern = checkerboard(1.0f, 13.0f, Int3(2, 2, 2), Int3(3, 3, 3));
	auto padded_view = paddedView(mirror(pattern, Bool3(false, true, false)), Int3(6, 6, 6), Int3(), 0.0f);
	auto rotated_view = rotatedView(pattern, rotationQuaternion(float(-kPi / 2), Float3(0.0f, 0.0f , 1.0f)), 0.5f * pattern.size(), Int3(6, 6, 6));

	testViewsForIdentity(padded_view, rotated_view);
}


BOLT_AUTO_TEST_CASE(MeshGridTest) {
	auto grid = meshGrid(Int3(), Int3(4, 4, 4));
	DeviceImage<int, 3> device_image(grid[0].size());
	HostImage<int, 3> host_image(grid[0].size());
	auto host_view = host_image.constView();
	for (int dim = 0; dim < 3; ++dim) {
		copy(grid[dim], device_image.view());
		copy(device_image.constView(), host_image.view());
		for (int k = 0; k < 4; ++k) {
			for (int j = 0; j < 4; ++j) {
				for (int i = 0; i < 4; ++i) {
					Int3 coords(i, j, k);
					BOOST_CHECK_EQUAL(host_view[coords], coords[dim]);
				}
			}
		}
	}
}

// Subview of hybrid image is not supported
BOLT_AUTO_TEST_CASE(SubviewOnDevice, BOLT_TEST_SKIP) {
	auto meshgrid = meshGrid(Int3(), Int3(8, 8, 8));
	auto small_meshgrid = meshGrid(Int3(2, 2, 2), Int3(6, 6, 6));

	for (int i = 0; i < 3; ++i) {
		auto mesh_subview = subview(meshgrid[i], Int3(2, 2, 2), Int3(4, 4, 4));
		DeviceImage<int, 3> device_subimage(4, 4, 4);
		copy(mesh_subview, device_subimage.view());

		BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_subview, small_meshgrid[i]), 0);
		BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_subimage.constView(), small_meshgrid[i]), 0);
	}
}

// Subview of hybrid image is not supported
BOLT_AUTO_TEST_CASE(CopySubviewOnDevice, BOLT_TEST_SKIP) {
	auto pattern = checkerboard(0, 1, Int2(4, 4), Int2(8, 8));
	DeviceImage<int, 2> device_image(8, 8);
	copy(pattern, device_image.view());

	auto subpattern = subview(pattern, Int2(2, 2), Int2(4, 4));
	auto small_pattern = checkerboard(0, 1, Int2(2, 2), Int2(4, 4));

	DeviceImage<int, 2> device_subimage(4, 4);
	copy(subview(device_image.constView(), Int2(2, 2), Int2(4, 4)), device_subimage.view());
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(small_pattern, device_subimage.constView()), 0);
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(small_pattern, subpattern), 0);
}


BOLT_AUTO_TEST_CASE(CopySubviewToHost) {
	auto pattern = checkerboard(1, 2, Int2(4, 4), Int2(8, 8));
	DeviceImage<int, 2> device_image(8, 8);
	copy(pattern, device_image.view());


	HostImage<int, 2> host_image(8, 8);
	host_image.clear();

	copy(
		subview(device_image.constView(), Int2(2, 2), Int2(4, 4)),
		subview(host_image.constView(), Int2(4, 4), Int2(4, 4)));

	auto host_view = host_image.constView();
	for (int j = 0; j < 8; ++j) {
		for (int i = 0; i < 8; ++i) {
			if (i < 4 || j < 4) {
				BOOST_CHECK_EQUAL(host_view[Int2(i, j)], 0);
			} else {
				BOOST_CHECK_EQUAL(host_view[Int2(i, j)], sum(div(Int2(i, j), Int2(2, 2))) % 2 ? 1 : 2);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(CopySubviewToHost3D) {
	DeviceImage<int, 3> device_image(32, 32, 32);
	copy(makeConstantImageView(int(5), device_image.size()), device_image.view());

	HostImage<int, 3> host_image(32, 32, 32);
	host_image.clear();
	copy(
		subview(device_image.constView(), Int3(), Int3(9, 17, 4)),
		subview(host_image.constView(), Int3(), Int3(9, 17, 4)));

	auto host_view = host_image.constView();
	for (int k = 0; k < host_view.size()[2]; ++k) {
		for (int j = 0; j < host_view.size()[1]; ++j) {
			for (int i = 0; i < host_view.size()[0]; ++i) {
				if (i < 9 && j < 17 && k < 4) {
					BOOST_CHECK_EQUAL(host_view[Int3(i, j, k)], 5);
				} else {
					BOOST_CHECK_EQUAL(host_view[Int3(i, j, k)], 0);
				}
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(CastView) {
	HostImage<int, 2> host_image(4, 4);
	auto host_view = host_image.view();

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			host_view[Int2(i, j)] = i + j;
		}
	}

	auto char_view = cast<char>(host_view);

	for (int j = 0; j < 4; ++j) {
		for (int i = 0; i < 4; ++i) {
			BOOST_CHECK_EQUAL(char_view[Int2(i, j)], i + j);
		}
	}
}

BOLT_AUTO_TEST_CASE(CopySliceOnDevice) {
	auto pattern = checkerboard(0, 1, Int3(4, 4, 4), Int3(8, 8, 8));
	DeviceImage<int, 3> device_image(8, 8, 8);
	copy(pattern, device_image.view());

	auto pattern_slice = slice<1>(pattern, 1);
	auto pattern_2d = checkerboard(0, 1, Int2(4, 4), Int2(8, 8));

	DeviceImage<int, 2> device_slice_image(8, 8);
	copy(slice<1>(device_image.constView(), 1), device_slice_image.view());

	BOOST_CHECK_EQUAL(squareDifferenceOfImages(pattern_2d, device_slice_image.constView()), 0);
	BOOST_CHECK_EQUAL(squareDifferenceOfImages(pattern_2d, pattern_slice), 0);
}


BOLT_AUTO_TEST_CASE(SliceOnDevice) {
	// meshgrid is used to generate 3 axis oriented image gradients
	// Slice in all 3 directions and check if we get correct gradient
	// or constant image in case of cut perpendicular to the gradient direction.
	auto meshgrid = meshGrid(Int3(), Int3(8, 8, 8));
	auto slice_meshgrid = meshGrid(Int2(), Int2(8, 8));

	for (int i = 0; i < 3; ++i) {
		auto mesh_slice = slice<1>(meshgrid[i], 3);
		DeviceImage<int, 2> device_slice_image(8, 8);
		copy(mesh_slice, device_slice_image.view());

		if (i != 1) {
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_slice, slice_meshgrid[std::min(i, 1)]), 0);
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_slice_image.constView(), slice_meshgrid[std::min(i, 1)]), 0);
		} else {
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(mesh_slice, makeConstantImageView(3, mesh_slice.size())), 0);
			BOOST_CHECK_EQUAL(squareDifferenceOfImages(device_slice_image.constView(), makeConstantImageView(3, mesh_slice.size())), 0);
		}
	}
}

BOLT_AUTO_TEST_CASE(ViewSizeException) {
	Int2 size2_a, size2_b(1, 2);
	bool did_throw = false;
	try {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(size2_a, size2_b));
	} catch (boost::exception &e) {
		// std::string const *extra  = boost::get_error_info<tag_index_view_pair_sizes>(e);
		std::pair<Int2, Int2> const *extra = boost::get_error_info<ViewPairSizesErrorInfoInt2>(e);
		BOOST_CHECK_EQUAL(extra->first, size2_a);
		BOOST_CHECK_EQUAL(extra->second, size2_b);
		did_throw = true;
	}
	BOOST_CHECK(did_throw);
}

BOLT_AUTO_TEST_CASE(MinMaxTest) {
	const Int2 testDimensions(16, 16);
	std::array<int, 16 * 16> testData;
	testData.fill(5);
	auto host_view = makeHostImageView(testData.data(), testDimensions);
	host_view[Int2(4, 4)] = 1;
	host_view[Int2(15, 0)] = 10;
	DeviceImage<int, 2> device_image(host_view.size());
	copy(host_view, device_image.view());
	Int2 min_max = minMax(device_image.constView());
	BOOST_CHECK_EQUAL(1, min_max[0]);
	BOOST_CHECK_EQUAL(10, min_max[1]);
}

BOLT_AUTO_TEST_CASE(MinMaxMeanTest) {
	const Int2 testDimensions(16, 16);
	std::array<int, 16 * 16> testData;
	testData.fill(5);
	auto host_view = makeHostImageView(testData.data(), testDimensions);
	host_view[Int2(4, 4)] = 1;
	host_view[Int2(15, 0)] = 10;
	DeviceImage<int, 2> device_image(host_view.size());
	copy(host_view, device_image.view());
	auto min_max_mean = minMaxMean(device_image.constView());
	BOOST_CHECK_EQUAL(1, min_max_mean.template get<0>());
	BOOST_CHECK_EQUAL(10, min_max_mean.template get<1>());
	BOOST_CHECK_EQUAL(5.00390625, min_max_mean.template get<2>());
}

BOLT_AUTO_TEST_CASE(ElementPositionPairAssignment) {
	ElementPositionPair<int, int> value(5, 0);

	int value_for_reference{0};
	ElementPositionPair<int &, int> reference(value_for_reference, 0);

	reference = value;
	BOOST_CHECK_EQUAL(value_for_reference, 5);
}

}  // namespace bolt
