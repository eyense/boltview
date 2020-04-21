// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#define BOOST_TEST_MODULE ConvolutionTest

#include <boltview/convolution.h>
#include <boltview/detail/convolution_foreach.h>
#include <tests/test_utils.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/transform.h>
#include <boltview/copy.h>
#include <boltview/procedural_views.h>
#include <boltview/loop_utils.h>
#include <boltview/create_view.h>

#ifdef BOLT_USE_UNIFIED_MEMORY
#include <boltview/unified_image.h>
#endif

#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boltview/image_io.h>

#include <chrono>


namespace bolt {

/// Test 2D kernel that just multiply
template<int tValue>
class MultiplyKernel {
public:
	static const bool kIsDynamicallyAllocated = false;
	static const bool kIsHostKernel = true;
	static const bool kIsDeviceKernel = true;
	static const int kDimension = 2;


	BOLT_DECL_HYBRID
	MultiplyKernel():
		size_(Int2(3, 3)), center_(Int2(1, 1))
	{}

	BOLT_DECL_HYBRID
	int operator[](Vector<int, 2> index) const{
		return kernel_[index[0]+center_[0] + size_[0] * (index[1]+center_[1])];
	}

	BOLT_DECL_HYBRID
	Vector<int, 2> size() const{
		return size_;
	}

	BOLT_DECL_HYBRID
	Vector<int, 2> center() const{
		return center_;
	}

private:
	Vector<int, 2> size_;
	Vector<int, 2> center_;

	int kernel_[9] = {
		0, 0, 0,
		0, tValue, 0,
		0, 0, 0};
};


// TODO(honza) Tests for gaussian and sobel

// Sum of all neighbors * 3, on host
BOLT_AUTO_TEST_CASE(ConstKernelTest) {
	const int multiplier = 3;
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = ConstantImageView<int,2>{1, Int2(8, 8)};
	auto out_view = image_out.view();

	convolution(checker_board, out_view, ConstKernel<int, 2>(multiplier, Int2(3, 3), Int2(1, 1)));

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(out_view[Int2(i, j)], 9 * multiplier);
		}
	}
}

// Sum of all neighbors * 3, on host
BOLT_AUTO_TEST_CASE(ConstKernelTestForeach) {
	const int multiplier = 3;
	HostImage<int, 2> image_in{8, 8};
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = ConstantImageView<int,2>{ 1, Int2(8, 8)};
	copy(checker_board, image_in.view());
	auto out_view = image_out.view();

	convolutionForeach(image_in.constView(), out_view, ConstKernel<int, 2>(multiplier, Int2(3, 3), Int2(1, 1)));

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(out_view[Int2(i, j)], 9 * multiplier);
		}
	}
}

// Sum of all neighbors * 3, on host
BOLT_AUTO_TEST_CASE(HostConstKernelTest) {
	const int multiplier = 3;
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = checkerboard(1, 3, Int2(2, 2), Int2(8, 8));
	auto out_view = image_out.view();

	convolution(checker_board, out_view, ConstKernel<int, 2>(multiplier, Int2(3, 3), Int2(1, 1)));

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int sum = 0;
				for(int k = -1; k <= 1; ++k){
					for(int l = -1; l <= 1; ++l){
						sum += checker_board[Int2(i+k, j+l)] * multiplier;
					}
				}
			BOOST_CHECK_EQUAL(out_view[Int2(i, j)], sum);
		}
	}
}

// Sum of all neighbors * 3, on host
BOLT_AUTO_TEST_CASE(HostConstKernelTestForeach) {
	const int multiplier = 3;
	HostImage<int, 2> image_out(8, 8);
	HostImage<int, 2> image_in(8, 8);

	auto checker_board = checkerboard(1, 3, Int2(2, 2), Int2(8, 8));
	copy(checker_board, image_in.view());
	auto out_view = image_out.view();

	convolutionForeach(image_in.constView(), out_view, ConstKernel<int, 2>(multiplier, Int2(3, 3), Int2(1, 1)));

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int sum = 0;
				for(int k = -1; k <= 1; ++k){
					for(int l = -1; l <= 1; ++l){
						sum += checker_board[Int2(i+k, j+l)] * multiplier;
					}
				}
			BOOST_CHECK_EQUAL(out_view[Int2(i, j)], sum);
		}
	}
}

// Sum of all neighbors * 3, on device
BOLT_AUTO_TEST_CASE(DeviceConstKernelTest) {
	DeviceImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = checkerboard(3, 2, Int2(2, 2), Int2(8, 8));

	convolution(checker_board, image.view(), ConstKernel<int, 2>(1, Int2(3, 3), Int2(1, 1)));

	copy(image.constView(), image_out.view());

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int sum = 0;
				for(int k = -1; k <= 1; ++k){
					for(int l = -1; l <= 1; ++l){
						sum += checker_board[Int2(i+k, j+l)];
					}
				}
			BOOST_CHECK_EQUAL(image_out.constView()[Int2(i, j)], sum);
		}
	}
}

// Sum of all neighbors * 3, on device
BOLT_AUTO_TEST_CASE(DeviceConstKernelTestForeach) {
	DeviceImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = checkerboard(3, 2, Int2(2, 2), Int2(8, 8));

	convolutionForeach(checker_board, image.view(), ConstKernel<int, 2>(1, Int2(3, 3), Int2(1, 1)));

	copy(image.constView(), image_out.view());

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int sum = 0;
				for(int k = -1; k <= 1; ++k){
					for(int l = -1; l <= 1; ++l){
						sum += checker_board[Int2(i+k, j+l)];
					}
				}
			BOOST_CHECK_EQUAL(image_out.constView()[Int2(i, j)], sum);
		}
	}
}

BOLT_AUTO_TEST_CASE(DeviceMultiplyProductKernelTest) {
	const int multiplier = 5;

	DeviceImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);


	auto checker_board = checkerboard(1, 0,  Int2(2, 2), Int2(8, 8));

	convolution(checker_board, image.view(), MultiplyKernel<multiplier>());

	copy(image.constView(), image_out.view());

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image_out.constView()[Int2(i, j)], multiplier * checker_board[Int2(i, j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DeviceMultiplyProductKernelTestForeach) {
	const int multiplier = 5;

	DeviceImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);


	auto checker_board = checkerboard(1, 0,  Int2(2, 2), Int2(8, 8));

	convolutionForeach(checker_board, image.view(), MultiplyKernel<multiplier>());

	copy(image.constView(), image_out.view());

	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image_out.constView()[Int2(i, j)], multiplier * checker_board[Int2(i, j)]);
		}
	}
}


BOLT_AUTO_TEST_CASE(DynamicHostTest) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto checker_board = checkerboard(1, 0,  Int2(2, 2), Int2(8, 8));

	int mat[] = { 0, 0, 0,
								0, 1, 0,
								0, 0, 0 };
	convolution(checker_board, image.view(), DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));


	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image.constView()[Int2(i, j)], checker_board[Int2(i, j)]);
		}
	}

	{

		HostImage<int, 2> test_out1(32, 32);
		HostImage<int, 2> test_out2(test_out1.size());
		auto checker_board = checkerboard(1, 0,  Int2(3, 3), test_out1.size());

		int mat1[] = {
			0, 0, 0,
			0, 0, 0,
			0, 0, 1 };
		auto k1 = DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat1);
		int mat2[] = {
			0, 0, 0,
			0, 1, 0,
			0, 0, 0 };
		auto k2 = DynamicHostKernel<int, 2>(Int2(3, 3), Int2(0, 0), mat2);

		convolution(checker_board, view(test_out1), k1);
		convolution(checker_board, view(test_out2), k2);
		int diff = reduce(square(subtract(view(test_out1), view(test_out2))), 0, thrust::plus<int>());

		BOOST_CHECK_EQUAL(diff, 0);
	}
}

BOLT_AUTO_TEST_CASE(DynamicHostTestForeach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_in(8, 8);

	auto checker_board = checkerboard(1, 0,  Int2(2, 2), Int2(8, 8));
	copy(checker_board, image_in.view());

	int mat[] = { 0, 0, 0,
								0, 1, 0,
								0, 0, 0 };
	convolutionForeach(image_in.constView(), image.view(), DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));


	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image.constView()[Int2(i, j)], checker_board[Int2(i, j)]);
		}
	}
}


BOLT_AUTO_TEST_CASE(DynamicHostTest2) {
	HostImage<int, 2> image(8, 8);

	auto checker_board = checkerboard(1, 1, Int2(2, 2), Int2(8, 8));

	int mat[] = { 0, 2, 0,
								2, 0, 2,
								0, 2, 0 };
	convolution(checker_board, image.view(), DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));


	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image.constView()[Int2(i, j)], 8);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicHostTest2Foreach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_in(8, 8);

	auto checker_board = checkerboard(1, 1, Int2(2, 2), Int2(8, 8));
	copy(checker_board, image_in.view() );

	int mat[] = { 0, 2, 0,
								2, 0, 2,
								0, 2, 0 };
	convolutionForeach(image_in.constView(), image.view(), DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));


	for (int i = 0; i < 8; ++i) {
		for (int j = 0; j < 8; ++j) {
			BOOST_CHECK_EQUAL(image.constView()[Int2(i, j)], 8);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicHostTest3) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = { 0, 0, 5,
								0, 2, 0,
								0, 0, 3 };

	convolution(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int res = 5 * view[Int2(i+1, j-1)] + 2 * view[Int2(i, j)] + 3 * view[Int2(i+1, j+1)];
			BOOST_CHECK_EQUAL(res, view_out[Int2(i, j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicHostTest3Foreach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = { 0, 0, 5,
								0, 2, 0,
								0, 0, 3 };

	convolutionForeach(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 3), Int2(1, 1), mat));

	for (int i = 1; i < 7; ++i) {
		for (int j = 1; j < 7; ++j) {
			int res = 5 * view[Int2(i+1, j-1)] + 2 * view[Int2(i, j)] + 3 * view[Int2(i+1, j+1)];
			BOOST_CHECK_EQUAL(res, view_out[Int2(i, j)]);
		}
	}
}

// Sum of horizontal neighbors
BOLT_AUTO_TEST_CASE(DynamicHostTest4) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {1, 0, 1};

	convolution(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 1), Int2(1, 0), mat));
	for(int j = 0; j < 8; ++j){
		for(int i = 1; i < 7; ++i){
			int sum = view[Int2(i-1, j)] + view[Int2(i+1, j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

// Sum of horizontal neighbors
BOLT_AUTO_TEST_CASE(DynamicHostTest4Foreach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {1, 0, 1};

	convolutionForeach(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 1), Int2(1, 0), mat));
	for(int j = 0; j < 8; ++j){
		for(int i = 1; i < 7; ++i){
			int sum = view[Int2(i-1, j)] + view[Int2(i+1, j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}


BOLT_AUTO_TEST_CASE(DynamicHostTest5) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolution(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));
	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicHostTest5Foreach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolutionForeach(image.constView(), view_out, DynamicHostKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));
	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

#ifdef BOLT_USE_UNIFIED_MEMORY

// Unified kernel
BOLT_AUTO_TEST_CASE(DynamicUnifiedTest) {
	HostImage<int, 2> image(8, 8);
	UnifiedImage<int, 2> unified_image(8, 8);

	auto view = image.view();
	auto unified_view = unified_image.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolution(constView(view), unified_view, DynamicUnifiedKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));


	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, unified_view[Int2(i, j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicUnifiedTestForeach) {
	HostImage<int, 2> image(8, 8);
	UnifiedImage<int, 2> unified_image(8, 8);

	auto view = image.view();
	auto unified_view = unified_image.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolutionForeach(constView(view), unified_view, DynamicUnifiedKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));


	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, unified_view[Int2(i, j)]);
		}
	}
}
#endif // BOLT_USE_UNIFIED_MEMORY


// Device kernel
BOLT_AUTO_TEST_CASE(DynamicDeviceTest) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> host_out(8, 8);
	DeviceImage<int, 2> device_image(8, 8);
	DeviceImage<int, 2> device_out(8, 8);

	auto view = image.view();
	auto view_out = host_out.view();
	auto device_view = device_image.view();
	auto device_out_view = device_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}
	copy(view, device_view);

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolution(device_image.constView(), device_out_view, DynamicDeviceKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));

	copy(device_out_view, view_out);

	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

BOLT_AUTO_TEST_CASE(DynamicDeviceTestPerf, BOLT_TEST_SKIP) {
	int size =  10000;
	HostImage<int, 2> image(size, size);
	HostImage<int, 2> host_out(size, size);
	DeviceImage<int, 2> device_image(size, size);
	DeviceImage<int, 2> device_out(size, size);

	auto view = image.view();
	auto view_out = host_out.view();
	auto device_view = device_image.view();
	auto device_out_view = device_out.view();

	int repeat_count = 1000;

	std::vector<double> times;

	for ( int k = 0; k < repeat_count; ++k) {
		for(int i = 0; i < size*size; ++i){
			linearAccess(view, i) = i;
		}
		copy(view, device_view);

		int mat[] = {3, 0, 1,
								 0, 5, 2};

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		convolution(device_image.constView(), device_out_view, DynamicDeviceKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
		times.push_back(time_span.count() );
	}

	double mean = std::accumulate(times.begin(), times.end(), 0. )/repeat_count;
	double var = sqrt(std::accumulate(times.begin(), times.end(), 0., [mean](double sum, double x){return sum + (x-mean)*(x-mean);} )/(repeat_count - 1));

	std::cout << "It took me " << mean << " +/-" << var << " seconds.";
	std::cout << std::endl;

	copy(device_out_view, view_out);

	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}


BOLT_AUTO_TEST_CASE(DynamicDeviceTestForeach) {
	HostImage<int, 2> image(8, 8);
	HostImage<int, 2> host_out(8, 8);
	DeviceImage<int, 2> device_image(8, 8);
	DeviceImage<int, 2> device_out(8, 8);

	auto view = image.view();
	auto view_out = host_out.view();
	auto device_view = device_image.view();
	auto device_out_view = device_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}
	copy(view, device_view);

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolutionForeach(device_image.constView(), device_out_view, DynamicDeviceKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));

	copy(device_out_view, view_out);

	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

#ifdef BOLT_USE_UNIFIED_MEMORY
// Device kernel + unified image
BOLT_AUTO_TEST_CASE(UnifiedDeviceTest) {
	UnifiedImage<int, 2> image(8, 8);
	UnifiedImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolution(constView(view), view_out, DynamicDeviceKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));

	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}

// Device kernel + unified image
BOLT_AUTO_TEST_CASE(UnifiedDeviceTestForeach) {
	UnifiedImage<int, 2> image(8, 8);
	UnifiedImage<int, 2> image_out(8, 8);

	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, 0, 1,
							 0, 5, 2};

	convolutionForeach(constView(view), view_out, DynamicDeviceKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat));

	for(int i = 2; i < 8; ++i){
		for(int j = 1; j < 8; ++j){
			int sum = 3 * view[Int2(i-2, j-1)]
							+     view[Int2(i,   j-1)]
							+ 5 * view[Int2(i-1, j)]
							+ 2 * view[Int2(i,   j)];
			BOOST_CHECK_EQUAL(sum, view_out[Int2(i, j)]);
		}
	}
}
#endif // BOLT_USE_UNIFIED_MEMORY


// NOTE(fidli): used to be disabled
// also causes compilatiton error
/*   ArrayView  CANNOT CREATE ImageLocator ON *ArrayView */
BOLT_AUTO_TEST_CASE(DeviceArrayViewTest, BOLT_TEST_SKIP) {
	thrust::device_vector<int> test_vector(100);
	thrust::device_vector<int> result(100);
	thrust::sequence(test_vector.begin(), test_vector.end());

	// NOTE(fidli): this causes compilation error
	/* auto view = makeDeviceArrayView(test_vector);
	auto result_view = makeDeviceArrayView(result);

	int mat[] = {1,0, 1,1};

	convolution(view, result_view, DynamicDeviceKernel<int, 2>(Int2(4,1), Int2(1,0), mat));
	*/
}


// 3D test
BOLT_AUTO_TEST_CASE(ConstHost3DTest) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();
	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = 1;
	}

	convolution(image.constView(), view_out, ConstKernel<int, 3>(1, Int3(3, 3, 3), Int3(1, 1, 1)));

	for(int i = 0; i < 8*8*8; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), 27);
	}
}

// 3D test
BOLT_AUTO_TEST_CASE(ConstHost3DTestForeach) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();
	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = 1;
	}

	convolutionForeach(image.constView(), view_out, ConstKernel<int, 3>(1, Int3(3, 3, 3), Int3(1, 1, 1)));

	for(int i = 0; i < 8*8*8; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), 27);
	}
}

BOLT_AUTO_TEST_CASE(Host3DTestIdentity) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {0, 1, 0};

	convolution(image.constView(), view_out, DynamicHostKernel<int, 3>(Int3(1, 1, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 8; ++i){
		for(int j = 0; j < 8; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = view[Int3(i, j, k)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(Host3DTestIdentityForeach) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {0, 1, 0};

	convolutionForeach(image.constView(), view_out, DynamicHostKernel<int, 3>(Int3(1, 1, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 8; ++i){
		for(int j = 0; j < 8; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = view[Int3(i, j, k)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(Host3DTest) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, -5, 2};

	convolution(image.constView(), view_out, DynamicHostKernel<int, 3>(Int3(1, 1, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 8; ++i){
		for(int j = 0; j < 8; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = 3 * view[Int3(i, j, k-1)]
								- 5 * view[Int3(i, j, k)]
								+ 2 * view[Int3(i, j, k+1)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(Host3DTestForeach) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {3, -5, 2};

	convolutionForeach(image.constView(), view_out, DynamicHostKernel<int, 3>(Int3(1, 1, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 8; ++i){
		for(int j = 0; j < 8; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = 3 * view[Int3(i, j, k-1)]
								- 5 * view[Int3(i, j, k)]
								+ 2 * view[Int3(i, j, k+1)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}

#ifdef BOLT_USE_UNIFIED_MEMORY
// 2x2x3 kernel
BOLT_AUTO_TEST_CASE(Host3DTest2) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = { 2, 0,
								0, 0,

								0, 3,
								5, 0,

								9, 0,
								0, 0 };

	convolution(constView(view), view_out, DynamicUnifiedKernel<int, 3>(Int3(2, 2, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 7; ++i){
		for(int j = 0; j < 7; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = 2 * view[Int3(i, j, k-1)]
								+ 3 * view[Int3(i+1, j, k)]
								+ 5 * view[Int3(i, j+1, k)]
								+ 9 * view[Int3(i, j, k+1)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}

// 2x2x3 kernel
BOLT_AUTO_TEST_CASE(Host3DTest2Foreach) {
	HostImage<int, 3> image(8, 8, 8);
	HostImage<int, 3> image_out(8, 8, 8);
	auto view = image.view();
	auto view_out = image_out.view();

	for(int i = 0; i < 8*8*8; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = { 2, 0,
								0, 0,

								0, 3,
								5, 0,

								9, 0,
								0, 0 };

	convolutionForeach(constView(view), view_out, DynamicUnifiedKernel<int, 3>(Int3(2, 2, 3), Int3(0, 0, 1), mat));

	for(int i = 0; i < 7; ++i){
		for(int j = 0; j < 7; ++j){
			for(int k = 1; k < 7; ++k){
				int sum = 2 * view[Int3(i, j, k-1)]
								+ 3 * view[Int3(i+1, j, k)]
								+ 5 * view[Int3(i, j+1, k)]
								+ 9 * view[Int3(i, j, k+1)];
				BOOST_CHECK_EQUAL(view_out[(Int3(i, j, k))], sum);
			}
		}
	}
}
#endif // BOLT_USE_UNIFIED_MEMORY

// float test
BOLT_AUTO_TEST_CASE(HostFloatTest) {
	HostImage<float, 2> img(8, 8);
	HostImage<float, 2> img_out(8, 8);

	auto view = img.view();
	auto view_out = img_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(view, i) = i;
	}

	float mat[] =  {0, 0,   0,
									0, 1.3, 0,
									0, 0,   0};
	convolution(img.constView(), view_out, DynamicHostKernel<float, 2>(Int2(3, 3), Int2(1, 1), mat));

	for(int i = 0; i < 8*8; ++i){
		BOOST_CHECK_CLOSE(linearAccess(view_out, i), linearAccess(view, i)*1.3, 0.0001);
	}
}

// float test
BOLT_AUTO_TEST_CASE(HostFloatTestForeach) {
	HostImage<float, 2> img(8, 8);
	HostImage<float, 2> img_out(8, 8);

	auto view = img.constView();
	auto view_out = img_out.view();

	for(int i = 0; i < 8*8; ++i){
		linearAccess(img.view(), i) = i;
	}

	float mat[] =  {0, 0,   0,
									0, 1.3, 0,
									0, 0,   0};
	convolutionForeach(view, view_out, DynamicHostKernel<float, 2>(Int2(3, 3), Int2(1, 1), mat));

	for(int i = 0; i < 8*8; ++i){
		BOOST_CHECK_CLOSE(linearAccess(view_out, i), linearAccess(view, i)*1.3, 0.0001);
	}
}

BOLT_AUTO_TEST_CASE(SharedMemory) {
	Int3 size(100, 100, 100);

	HostImage<float, 3> image(size);
	HostImage<float, 3> image_out(size);
	HostImage<float, 3> image_out_shared(size);

	DeviceImage<float, 3> device_image(size);
	DeviceImage<float, 3> device_image_out(size);
	DeviceImage<float, 3> device_image_out_shared(size);

	auto device_view = device_image.constView();
	auto device_view_out = device_image_out.view();
	auto device_view_out_shared = device_image_out_shared.view();

	for (int i = 0; i < product(size); ++i){
		linearAccess(image.view(), i) = i;
	}
	copy(image.view(), device_image.view());

	float mat[] = {1, 1, 1, 1, 1, 1, 1, 1};
	Vector<int, 3> kernel_size(2, 2, 2);
	Vector<int, 3> kernel_center(1, 1, 1);

	convolution(
		device_view,
		device_view_out_shared,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, true>(device_view, device_view_out)
	);

	convolution(
		device_view,
		device_view_out,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, false>(device_view, device_view_out)
	);


	copy(device_view_out, image_out.view());
	copy(device_view_out_shared, image_out_shared.view());

	for (int i = 0; i < product(size); ++i){
		BOOST_CHECK_EQUAL(linearAccess(image_out_shared.view(), i), linearAccess(image_out.view(), i));
	}

	BOLT_CHECK_ERROR_STATE("Shared Memory convolution");
}

BOLT_AUTO_TEST_CASE(SharedMemoryForeach) {
	Int3 size(100, 100, 100);

	HostImage<float, 3> image(size);
	HostImage<float, 3> image_out(size);
	HostImage<float, 3> image_out_shared(size);

	DeviceImage<float, 3> device_image(size);
	DeviceImage<float, 3> device_image_out(size);
	DeviceImage<float, 3> device_image_out_shared(size);

	auto device_view = device_image.constView();
	auto device_view_out = device_image_out.view();
	auto device_view_out_shared = device_image_out_shared.view();

	for (int i = 0; i < product(size); ++i){
		linearAccess(image.view(), i) = i;
	}
	copy(image.view(), device_image.view());

	float mat[] = {1, 1, 1, 1, 1, 1, 1, 1};
	Vector<int, 3> kernel_size(2, 2, 2);
	Vector<int, 3> kernel_center(1, 1, 1);

	convolutionForeach(
		device_view,
		device_view_out_shared,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, true>(device_view, device_view_out)
	);

	convolutionForeach(
		device_view,
		device_view_out,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, false>(device_view, device_view_out)
	);


	copy(device_view_out, image_out.view());
	copy(device_view_out_shared, image_out_shared.view());

	for (int i = 0; i < product(size); ++i){
		BOOST_CHECK_EQUAL(linearAccess(image_out_shared.view(), i), linearAccess(image_out.view(), i));
	}

	BOLT_CHECK_ERROR_STATE("Shared Memory convolution");
}

BOLT_AUTO_TEST_CASE(SharedMemorySmall) {
	Int3 size(10, 10, 10);

	HostImage<float, 3> image(size);
	HostImage<float, 3> image_out(size);
	HostImage<float, 3> image_out_shared(size);

	DeviceImage<float, 3> device_image(size);
	DeviceImage<float, 3> device_image_out(size);
	DeviceImage<float, 3> device_image_out_shared(size);

	auto device_view = device_image.constView();
	auto device_view_out = device_image_out.view();
	auto device_view_out_shared = device_image_out_shared.view();

	for (int i = 0; i < product(size); ++i){
		linearAccess(image.view(), i) = i;
	}
	copy(image.view(), device_image.view());

	float mat[] = {1, 1, 1, 1, 1, 1, 1, 1};
	Vector<int, 3> kernel_size(2, 2, 2);
	Vector<int, 3> kernel_center(1, 1, 1);

	convolution(
		device_view,
		device_view_out_shared,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, true>(device_view, device_view_out)
	);

	convolution(
		device_view,
		device_view_out,
		DynamicDeviceKernel<float, 3>(kernel_size, kernel_center, mat),
		getDefaultConvolutionPolicy<decltype(device_view), decltype(device_view_out), BorderHandling::kNone, false>(device_view, device_view_out)
	);


	copy(device_view_out, image_out.view());
	copy(device_view_out_shared, image_out_shared.view());

	/*std::cout << Dump (image_out.view(), "test1") << std::endl;
	std::cout << Dump (image_out_shared.view(), "test2") << std::endl;*/

	for (int i = 0; i < product(size); ++i){
		BOOST_CHECK_EQUAL(linearAccess(image_out_shared.view(), i), linearAccess(image_out.view(), i));
	}

	BOLT_CHECK_ERROR_STATE("Shared Memory convolution");
}

#ifdef BOLT_USE_UNIFIED_MEMORY

BOLT_AUTO_TEST_CASE(SharedMemoryUnified) {
	const int SIZEX = 32;
	const int SIZEY = 16;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									0, 0, 0,
									0, 0, 1};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(2, 2), mat);

	convolution(constView(view), view_out, kernel);
	convolution(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));


	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemoryUnifiedForeach) {
	const int SIZEX = 32;
	const int SIZEY = 16;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									0, 0, 0,
									0, 0, 1};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(2, 2), mat);

	convolutionForeach(constView(view), view_out, kernel);
	convolutionForeach(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));


	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemory2) {
	const int SIZEX = 32;
	const int SIZEY = 16;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									1, 0, 0,
									0, 0, 0};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(0, 1), mat);

	convolution(constView(view), view_out, kernel);
	convolution(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));


	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemory2Foreach) {
	const int SIZEX = 32;
	const int SIZEY = 16;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									1, 0, 0,
									0, 0, 0};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(0, 1), mat);

	convolutionForeach(constView(view), view_out, kernel);
	convolutionForeach(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));


	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemoryLargerImage) {
	const int SIZEX = 1000;
	const int SIZEY = 100;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									0, 0, 0,
									0, 0, 1};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(2, 2), mat);

	convolution(constView(view), view_out, kernel);
	convolution(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));
	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemoryLargerImageForeach) {
	const int SIZEX = 1000;
	const int SIZEY = 100;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] =  {0, 0, 0,
									0, 0, 0,
									0, 0, 1};
	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 3), Int2(2, 2), mat);

	convolutionForeach(constView(view), view_out, kernel);
	convolutionForeach(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));

	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemory3D) {
	const int SIZEX = 100;
	const int SIZEY = 100;
	const int SIZEZ = 10;

	UnifiedImage<int, 3> img(SIZEX, SIZEY, SIZEZ);
	UnifiedImage<int, 3> img_out(SIZEX, SIZEY, SIZEZ);
	UnifiedImage<int, 3> img_out_shared(SIZEX, SIZEY, SIZEZ);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY*SIZEZ; ++i){
		linearAccess(view, i) = i;
	}

	auto kernel = getGaussian<3>(2);

	convolution(constView(view), view_out, kernel);
	convolution(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));
	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY*SIZEZ; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemory3DForeach) {
	const int SIZEX = 100;
	const int SIZEY = 100;
	const int SIZEZ = 10;

	UnifiedImage<int, 3> img(SIZEX, SIZEY, SIZEZ);
	UnifiedImage<int, 3> img_out(SIZEX, SIZEY, SIZEZ);
	UnifiedImage<int, 3> img_out_shared(SIZEX, SIZEY, SIZEZ);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY*SIZEZ; ++i){
		linearAccess(view, i) = i;
	}

	auto kernel = getGaussian<3>(2);

	convolutionForeach(constView(view), view_out, kernel);
	convolutionForeach(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));
	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY*SIZEZ; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemoryAsymmetric) {
	const int SIZEX = 100;
	const int SIZEY = 100;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {0, 1, 2,
							 3, 4, 5};

	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat);

	convolution(constView(view), view_out, kernel);
	convolution(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));
	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}

BOLT_AUTO_TEST_CASE(SharedMemoryAsymmetricForeach) {
	const int SIZEX = 100;
	const int SIZEY = 100;

	UnifiedImage<int, 2> img(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<int, 2> img_out_shared(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out_shared = img_out_shared.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	int mat[] = {0, 1, 2,
							 3, 4, 5};

	auto kernel = DynamicUnifiedKernel<int, 2>(Int2(3, 2), Int2(2, 1), mat);

	convolutionForeach(constView(view), view_out, kernel);
	convolutionForeach(
		constView(view),
		view_out_shared,
		kernel,
		getDefaultConvolutionPolicy(view, view_out_shared));

	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		BOOST_CHECK_EQUAL(linearAccess(view_out, i), linearAccess(view_out_shared, i));
	}
}


BOLT_AUTO_TEST_CASE(Gauss) {
	const int SIZEX = 100;
	const int SIZEY = 100;

	UnifiedImage<float, 2> img(SIZEX, SIZEY);
	UnifiedImage<float, 2> img_out(SIZEX, SIZEY);
	UnifiedImage<float, 2> img_out2(SIZEX, SIZEY);
	UnifiedImage<float, 2> tmp(SIZEX, SIZEY);

	auto view = img.view();
	auto view_out = img_out.view();
	auto view_out2 = img_out2.view();
	auto view_tmp = tmp.view();

	for(int i = 0; i < SIZEX*SIZEY; ++i){
		linearAccess(view, i) = i;
	}

	auto kernel = getGaussian<2>(2);
	auto kernel_sep = getSeparableGaussian<2>(2);
	convolution(constView(view), view_out, kernel);
	separableConvolution(
		constView(view),
		view_out2,
		view_tmp,
		kernel_sep,
		getDefaultConvolutionPolicy(view, view_out2));

	BOLT_CHECK(cudaDeviceSynchronize());
	for(int i = 0; i < SIZEX*SIZEY; ++i){
		// Should it be more precise? Eg. 0.0001?
		BOOST_CHECK_CLOSE(linearAccess(view_out, i), linearAccess(view_out2, i), 0.001);
	}
}

#endif // BOLT_USE_UNIFIED_MEMORY

}  // namespace bolt
