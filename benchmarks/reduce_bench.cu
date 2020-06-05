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
#include <boltview/reduce.h>
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

	BenchmarkManager bm;

	for (int i : { 1, 4, 6, 8, 12, 16 }) {
		bm.add("Normal bucket" + std::to_string(i), [=](Timer & timer){
			DeviceImage<float, 3> test_image(kImageSize);
			auto in_view = view(test_image);

			auto cb = checkerboard(10, 2, Int3(8, 8, 8), kImageSize);
			copy(cb, in_view);

			{
				auto interval = timer.start("reduce");
				ExecutionPolicy policy;
				policy.bucket_size = i;
				auto red = Reduce<float, true>(cb.size(), policy);
				for (int i = 0; i < kIterationCount; ++i) {
					// auto v = sum(in_view, 3.3f, policy);
					float v;
					red.runAsync(in_view, v, 3.3f, thrust::plus<float>());
				}
				cudaDeviceSynchronize();
			}
		});
	}

	auto switched_strides = [=](Timer & timer){
		DeviceImage<float, 3> test_image(kImageSize);
		auto original_view = view(test_image);

		auto new_size = swizzle<2, 1, 0>(original_view.size());
		auto new_strides = swizzle<2, 1, 0>(original_view.strides());

		auto in_view = makeDeviceImageView(original_view.pointer(), new_size, new_strides);

		auto cb = checkerboard(10, 2, Int3(8, 8, 8), kImageSize);

		copy(cb, in_view);

		{
			auto interval = timer.start("reduce");
			for (int i = 0; i < kIterationCount; ++i) {
				auto v = sum(in_view, 3.3f);
			}
			cudaDeviceSynchronize();
		}

	};


	// bm.add("Switched strides", switched_strides);

	bm.add("Dimension reduce", [=](Timer & timer){
		DeviceImage<float, 3> test_image(kImageSize);
		DeviceImage<float, 2> reduce_image(removeDimension(kImageSize, 0));
		auto in_view = view(test_image);
		auto out_view = view(reduce_image);

		auto cb = checkerboard(10, 2, Int3(8, 8, 8), kImageSize);
		copy(cb, in_view);

		{
			auto interval = timer.start("reduce");
			ExecutionPolicy policy;
			auto red = DimensionReduce<float, 3, 0, true>(cb.size(), policy);
			for (int i = 0; i < kIterationCount; ++i) {
				red.runAsync(in_view, out_view, 0.0f, thrust::plus<float>());
			}
			cudaDeviceSynchronize();
		}
	});

	bm.runAll();
	bm.printAll();

	return 0;
}
