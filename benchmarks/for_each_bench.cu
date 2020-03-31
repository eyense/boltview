// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#include <unistd.h>

#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/unified_image.h>
#include <boltview/procedural_views.h>
#include <boltview/create_view.h>
#include <boltview/subview.h>
#include <boltview/for_each.h>
#include <boltview/convolution.h>
#include <boltview/convolution_kernels.h>
#include <boltview/transform.h>
#include <boltview/copy.h>


#include "benchmarks.h"

using namespace bolt;
using namespace std;

struct TestFunctor {
	BOLT_DECL_DEVICE
	void operator()(float &val) const {
		auto tmp = log(2 + sin(val));
		val = tmp;
	}

};


int main() {
	const Int3 kImageSize(600, 600, 300);
	const int kIterationCount = 200;

	auto normal = [=](Timer & timer){
		DeviceImage<float, 3> test_image(kImageSize);
		auto in_view = view(test_image);

		auto cb = checkerboard(10, 2, Int3(80, 80, 80), kImageSize);

		copy(cb, in_view);

		{
			auto foreach_interval = timer.start("foreach");
			for (int i = 0; i < kIterationCount; ++i) {
				forEach(in_view, TestFunctor{});
			}
			cudaDeviceSynchronize();
		}

	};

	auto switched_strides = [=](Timer & timer){
		DeviceImage<float, 3> test_image(kImageSize);
		auto original_view = view(test_image);

		auto new_size = swizzle<2, 1, 0>(original_view.size());
		auto new_strides = swizzle<2, 1, 0>(original_view.strides());

		auto in_view = makeDeviceImageView(original_view.pointer(), new_size, new_strides);

		auto cb = checkerboard(10, 2, Int3(80, 80, 80), kImageSize);

		copy(cb, in_view);

		{
			auto foreach_interval = timer.start("foreach");
			for (int i = 0; i < kIterationCount; ++i) {
				forEach(in_view, TestFunctor{});
			}
			cudaDeviceSynchronize();
		}

	};


	BenchmarkManager bm;
	bm.add("Normal", normal);
	bm.add("Switched strides", switched_strides);

	bm.runAll();
	bm.printAll();

	return 0;
}
