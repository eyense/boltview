// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#define BOOST_TEST_MODULE TextureImageTest
#include <boost/test/included/unit_test.hpp>
#include <boltview/tests/test_utils.h>
#include <cuda.h>
#include <boltview/copy.h>
#include <boltview/detail/meta_algorithm_utils.h>
#include <boltview/host_image.h>
#include <boltview/host_image_view.h>
#include <boltview/device_image.h>
#include <boltview/procedural_views.h>
#include <boltview/subview.h>
#include <boltview/texture_image.h>
#include <boltview/tests/texture_image_test_utils.h>

#include <algorithm>
#include <boost/timer.hpp>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace bolt {

// using old global reference texturing API for comparison
static texture<float, 3, cudaReadModeElementType> texture_ref;

void bindToProjTexture(cudaArray* cuda_array) {
	texture_ref.addressMode[0] = cudaAddressModeClamp;
	texture_ref.addressMode[1] = cudaAddressModeClamp;
	texture_ref.addressMode[2] = cudaAddressModeClamp;
	texture_ref.filterMode = cudaFilterModeLinear;
	texture_ref.normalized = 0;

	BOLT_CHECK(cudaBindTextureToArray(texture_ref, cuda_array, cudaCreateChannelDesc<float>()));
}

void UnbindProjTexture() {
	BOLT_CHECK(cudaUnbindTexture(texture_ref));
}

template<typename TTextureView, typename TView>
BOLT_GLOBAL void interpolationTestKernel(TTextureView texture_view, TView output, Float3 offset) {
	Int3 object_coords = bolt::MapBlockIdxAndThreadIdxToViewCoordinates<3>();

	if (texture_view.IsIndexInside(object_coords)) {
		Float3 projection_coords(object_coords);
		projection_coords += offset;

		float bolt_value = texture_view.access(projection_coords);
		float fastcone_value = tex3D(texture_ref, projection_coords[0] + 0.5f, projection_coords[1] + 0.5f, projection_coords[2] + 0.5f);

		output[object_coords] = abs(bolt_value - fastcone_value);
	}
}

template<typename TTextureView, typename TDeviceView>
void runInterpolationTestKernel(TTextureView texture_view, TDeviceView output_view, Float3 offset) {
	dim3 blockSize = detail::DefaultBlockDimForDimension<3>();
	dim3 gridSize = detail::DefaultGridSizeForBlockDim(texture_view.size(), blockSize);

	bindToProjTexture(texture_view.CudaArray());
	interpolationTestKernel<<<gridSize, blockSize>>>(texture_view, output_view, offset);
	BOLT_CHECK(cudaThreadSynchronize());
	UnbindProjTexture();
}

// NOTE(fidli): test used to be disabled
// compare TextureImage interpolation vs the old texture reference interpolation
BOLT_AUTO_TEST_CASE(TestTextureObjectVsTextureReference, BOLT_TEST_SKIP) {
	Int3 texture_size(64, 128, 256);
	TextureImage<float, 3> texture_image(texture_size);
	DeviceImage<float, 3> output(texture_size);

	HostImage<float, 3> random_data(texture_size);
	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());

	Float3 offsets[] = {
		Float3(0.0f, 0.0f, 0.0f),
		Float3(0.5f, 0.5f, 0.0f),
		Float3(0.5f, 0.5f, 0.5f),
		Float3(0.5f, -0.7f, 1.5f),
		Float3(-4.5f, -15.7f, -3.0f)
	};

	for (Float3 offset : offsets) {
		runInterpolationTestKernel(texture_image.view(), output.view(), offset);
		std::cout << "Tested with offset " << offset[0] << ", " << offset[1] << std::endl;
		checkViewCloseToZero(output.view());
	}
}

template<typename TTextureView, typename TView>
BOLT_GLOBAL void elementAccessTestKernel(TTextureView texture_view, TView device_view, TView output) {
	typename TView::IndexType object_coords = MapBlockIdxAndThreadIdxToViewCoordinates<TView::kDimension>();
	typename TTextureView::IndexType texture_coords(object_coords);

	if (device_view.IsIndexInside(object_coords)) {
		output[object_coords] = texture_view[texture_coords] - device_view[object_coords];
	}
}

template<typename TTextureView, typename TDeviceView>
void runElementAccessTestKernel(TTextureView texture_view, TDeviceView device_view, TDeviceView output_view){
	dim3 block_size = detail::DefaultBlockDimForDimension<TTextureView::kDimension>();
	dim3 grid_size = detail::DefaultGridSizeForBlockDim(device_view.size(), block_size);
	elementAccessTestKernel<<<grid_size, block_size>>>(texture_view, device_view, output_view);
	BOLT_CHECK(cudaThreadSynchronize());
}

// test whether accessing a TextureImage<TType, tDimension> at integer coordinates gives same values as DeviceImage<TType, tDimension>
template<typename TType, int tDimension>
void runElementAccessTestForDim(){
	auto texture_size = Vector<int, tDimension>::Fill(55);
	TextureImage<TType, tDimension> texture_image(texture_size);
	DeviceImage<TType, tDimension> device_image(texture_size);
	DeviceImage<TType, tDimension> output(texture_size);

	HostImage<TType, tDimension> random_data(texture_size);
	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());
	copy(random_data.view(), device_image.view());

	runElementAccessTestKernel(texture_image.view(), device_image.view(), output.view());
	checkViewCloseToZero(output.view());
}

// NOTE(fidli): test used to be disabled
// test whether accessing a TextureImage at integer coordinates gives same values as DeviceImage
BOLT_AUTO_TEST_CASE(TestTextureViewElementAccess, BOLT_TEST_SKIP)
{
	runElementAccessTestForDim<float, 2>();
	runElementAccessTestForDim<Float2, 2>();
	runElementAccessTestForDim<Float4, 2>();
	runElementAccessTestForDim<float, 3>();
	runElementAccessTestForDim<Float2, 3>();
	runElementAccessTestForDim<Float4, 3>();

	runElementAccessTestForDim<int, 2>();
	runElementAccessTestForDim<Int2, 2>();
	runElementAccessTestForDim<Int4, 2>();
	runElementAccessTestForDim<int, 3>();
	runElementAccessTestForDim<Int2, 3>();
	runElementAccessTestForDim<Int4, 3>();
}

template<typename TTextureView, typename TView>
BOLT_GLOBAL void CustomFilteringTestKernel(TTextureView texture_view, TView output) {
	Int2 object_coords = MapBlockIdxAndThreadIdxToViewCoordinates<2>();
	Float2 texture_coords(object_coords);

	texture_coords[0] = (texture_coords[0] - texture_view.size()[0] / 2.0f + 0.5f) / texture_view.size()[0];
	texture_coords[1] = (texture_coords[1] - texture_view.size()[1] / 2.0f + 0.5f) / texture_view.size()[1];

	output[object_coords] = texture_view.access(texture_coords);
}

template<typename TTextureView, typename TDeviceView>
void RunCustomFilteringTestKernel(TTextureView texture_view, TDeviceView output_view) {
	dim3 block_size(16, 16);
	dim3 grid_size = detail::DefaultGridSizeForBlockDim(texture_view.size(), block_size);
	CustomFilteringTestKernel<<<grid_size, block_size>>>(texture_view, output_view);
	BOLT_CHECK(cudaThreadSynchronize());
}

// use normalized coordinates and bordered access in x coordinate, rest of parameters are inherited from CudaTypeDefaults
struct CustomFilteringNormalizedBorderedInX : public CudaType<float> {
	static const bool normalized_coords = true;
	static const cudaTextureAddressMode address_mode_x = cudaAddressModeBorder;
};

// NOTE(fidli): test used to be disabled
// test a custom filtering type for TextureImage, this also serves as a how-to for custom filtering types
BOLT_AUTO_TEST_CASE(CustomFilteringTypeTest, BOLT_TEST_SKIP) {
	Int2 texture_size(4, 6);
	dim3 block_size(16, 16);

	TextureImage<float, 2, CustomFilteringNormalizedBorderedInX> texture_image(texture_size);
	DeviceImage<float, 2> output(Int2(block_size.x, block_size.y));

	HostImage<float, 2> output_host(Int2(block_size.x, block_size.y));
	HostImage<float, 2> random_data(texture_size);
	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());

	RunCustomFilteringTestKernel(texture_image.view(), output.view());
	copy(output.view(), output_host.view());

	for (int j = 0; j < static_cast<int>(block_size.y); j++) {
		for (int i = 0; i < static_cast<int>(block_size.x); i++) {
			Int2 coords(i, j);
			if (i >= texture_size[0]) {
				BOOST_CHECK_SMALL(output_host.view()[coords], std::numeric_limits<float>::epsilon());
			} else {
				Int2 source_coords = Int2(i, std::min(texture_size[1] - 1, j));
				BOOST_CHECK_CLOSE(output_host.view()[coords], random_data.view()[source_coords], 0.0001);
			}
		}
	}
}

// NOTE(fidli): test used to be disabled
// test that a subview of a TextureImage gives same data as a subview of a DeviceImage
BOLT_AUTO_TEST_CASE(TextureSubviewTest, BOLT_TEST_SKIP) {
	Int2 texture_size(16, 16);
	Int2 subview_size(8, 12);
	Int2 corner(2, 4);

	TextureImage<float, 2> texture_image(texture_size);
	DeviceImage<float, 2> device_image(texture_size);
	HostImage<float, 2> random_data(texture_size);

	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());
	copy(random_data.view(), device_image.view());

	auto texture_subview = Subview(texture_image.view(), Int2(corner), subview_size);
	auto device_subview = Subview(device_image.view(), corner, subview_size);

	checkViewCloseToZero(Subtract(texture_subview, device_subview));
}

// NOTE(fidli): test used to be disabled
// test that a slice of a TextureImage gives same data as a slice of a DeviceImage
BOLT_AUTO_TEST_CASE(TextureSliceTest, BOLT_TEST_SKIP) {
	Int3 texture_size(16, 16, 16);
	const int sliceDim = 1;
	int slicePos = 4;

	TextureImage<float, 3> texture_image(texture_size);
	DeviceImage<float, 3> device_image(texture_size);
	HostImage<float, 3> random_data(texture_size);

	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());
	copy(random_data.view(), device_image.view());

	auto texture_slice = Slice<sliceDim>(texture_image.view(), slicePos);
	auto device_slice = Slice<sliceDim>(device_image.view(), slicePos);

	checkViewCloseToZero(Subtract(texture_slice, device_slice));
}


template<typename TView1, typename TView2>
BOLT_GLOBAL void TestKernel(TView1 view1, TView2 view2) {
	if (threadIdx.x == 0) {
		for (int i = 0; i < 22; ++i) {
			printf("%f ", view1[Int3(i, 0, 0)]);
		}
		printf("\n");
		for (int i = 0; i < 22; ++i) {
			printf("%f ", view2[Int3(i, 0, 0)]);
		}
		printf("\n");
	}
	__syncthreads();
}

// NOTE(fidli): disabled checks
BOLT_AUTO_TEST_CASE(TextureSubviewCopyTest, BOLT_TEST_SKIP) {
	Int3 texture_size(16, 16, 16);

	//auto checker_board = checkerboard(0.0f, 3.0f, Int3(8, 8, 8), Int3(24, 88, 56));
	auto checker_board = MakeUniqueIdImageView(Int3(24, 88, 56), 2.0f);

	HostImage<float, 3> device_image(Int3(30, 30, 30));
	TextureImage<float, 3> texture_image(Int3(22, 22, 22));

	auto checker_subview = Subview(checker_board, Int3(1, 2, 3), Int3(22, 22, 22));
	auto device_subview = Subview(device_image.view(), Int3(1, 2, 3), Int3(22, 22, 22));
	auto texture_subview = texture_image.view();//Subview(texture_image.view(), Int3()/*(1, 2, 3)*/, Int3(22, 22, 22));

	copy(checker_subview, device_subview);
	copy(device_subview, texture_subview);


	//checkViewCloseToZero(Subtract(device_subview, checker_subview));
	//checkViewCloseToZero(Subtract(texture_subview, checker_subview));

	dim3 block_size(16, 1, 1);
	dim3 grid_size(1, 1, 1);
	TestKernel<<<grid_size, block_size>>>(texture_subview, checker_subview);
	BOLT_CHECK(cudaThreadSynchronize());
}

// NOTE(fidli): used to be disabled
//TODO(johny) floting point offsets for subview and slice together with other interpolated image views
// test that a TextureImage subview works correctly with floating point offsets
BOLT_AUTO_TEST_CASE(TextureSubviewFloatingPointAccessTest, BOLT_TEST_SKIP) {
	Int3 texture_size(16, 16, 16);
	Int3 subview_size(12, 12, 12);

	const int sliceDim = 0;
	int slicePos = 4;

	TextureImage<float, 3> texture_image(texture_size);
	HostImage<float, 3> random_data(texture_size);

	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());

	auto texture_slice_1 = Slice<sliceDim>(texture_image.view(), slicePos);
	auto texture_subview_1 = Subview(texture_slice_1, Float2(0.5f, 0.5f), Int2(12, 12));

	auto texture_subview_2 = Subview(texture_image.view(), Int3(0, 0, 0), subview_size);
	auto texture_slice_2 = Slice<sliceDim>(texture_subview_2, slicePos);

	checkViewCloseToZero(Subtract(texture_slice_2, texture_subview_1));
}

// NOTE(fidli): used to be disabled
// test that a TextureImage slice works correctly with floating point offset
BOLT_AUTO_TEST_CASE(TextureSliceFloatingPointAccessTest, BOLT_TEST_SKIP) {
	Int3 texture_size(16, 16, 16);
	Int3 subview_size(12, 12, 12);

	const int sliceDim = 2;
	float slicePos = 0.5f;

	TextureImage<float, 3> texture_image(texture_size);
	HostImage<float, 3> random_data(texture_size);

	GenerateRandomView(random_data.view(), 1337ULL);
	copy(random_data.view(), texture_image.view());

	auto texture_slice_1 = Slice<sliceDim >(texture_image.view(), slicePos);
	auto texture_subview_1 = Subview(texture_slice_1, Int2(), Int2(12, 12));

	auto texture_subview_2 = Subview(texture_image.view(), Float3(0.0f, 0.0f, slicePos), subview_size);
	auto texture_slice_2 = Slice<sliceDim >(texture_subview_2, 0);

	checkViewCloseToZero(Subtract(texture_slice_2, texture_subview_1));
}

}  // namespace bolt
