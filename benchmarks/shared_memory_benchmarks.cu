// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#include <unistd.h>

#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/unified_image.h>
#include <boltview/procedural_views.h>
#include <boltview/create_view.h>
#include <boltview/subview.h>
#include <boltview/convolution.h>
#include <boltview/convolution_kernels.h>
#include <boltview/transform.h>
#include <boltview/copy.h>


#include "benchmarks.h"

using namespace bolt;
using namespace std;



int main() {
	const int sizeX = 1000;
	const int sizeY = 1000;
	const int sizeZ = 100;
	const float std_dev = 2.5;

	const int sizeX2D = 32500;
	const int sizeY2D = 32500;


	auto normal = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 2> in(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> out(sizeX2D, sizeY2D);
		auto in_view = view(in);
		auto out_view = view(out);

		auto cb = checkerboard(10, 2, Int2(sizeX2D/10, sizeY2D/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("copy");
			copy(cb, in_view);
		}

		auto kernel = getGaussian<2>(std_dev);

		{
			auto convolution_interval = timer.start("convolution");
			convolution(in_view, out_view, kernel);
			cudaDeviceSynchronize();
		}

	};

	auto shared_memory = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 2> in(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> out(sizeX2D, sizeY2D);
		auto in_view = view(in);
		auto out_view = view(out);

		auto cb = checkerboard(10, 2, Int2(sizeX2D/10, sizeY2D/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("copy");
			copy(cb, in_view);

		}
		auto kernel = getGaussian<2>(std_dev);

		{
			auto convolution_interval = timer.start("convolution");
			convolution(in_view,
					out_view,
					kernel,
					getDefaultConvolutionPolicy(in_view, out_view));
			cudaDeviceSynchronize();
		}

	};


	auto normal3D = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 3> in(sizeX, sizeY, sizeZ);
		UnifiedImage<float, 3> out(sizeX, sizeY, sizeZ);
		auto in_view = view(in);
		auto out_view = view(out);

		auto cb = checkerboard(10, 2, Int2(sizeX2D/10, sizeY2D/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("Init");
			for(int z = 0; z < sizeZ; ++z){
				auto sview = subview(in_view, Int3(0,0, z), Int3(sizeX, sizeY, 1));
				for(int i = 0; i < sizeX*sizeY; ++i){
					if(z % 20 < 10){
						linearAccess(sview, i) = linearAccess(cb, i);
					}
					else{
						linearAccess(sview, i) = 0;
					}
				}
			}
		}

		auto kernel = getGaussian<3>(std_dev);

		{
			auto convolution_interval = timer.start("convolution");
			convolution(in_view, out_view, kernel);
			cudaDeviceSynchronize();
		}

	};

	auto shared_memory3D = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 3> in(sizeX, sizeY, sizeZ);
		UnifiedImage<float, 3> out(sizeX, sizeY, sizeZ);
		auto in_view = view(in);
		auto out_view = view(out);

		auto cb = checkerboard(10, 2, Int2(sizeX2D/10, sizeY2D/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("Init");
			for(int z = 0; z < sizeZ; ++z){
				auto sview = subview(in_view, Int3(0,0, z), Int3(sizeX, sizeY, 1));
				for(int i = 0; i < sizeX * sizeY; ++i){
					if(z % 20 < 10){
						linearAccess(sview, i) = linearAccess(cb, i);
					}
					else{
						linearAccess(sview, i) = 0;
					}
				}
			}
		}

		auto kernel = getGaussian<3>(std_dev);

		{
			auto convolution_interval = timer.start("convolution");
			convolution(in_view,
					out_view,
					kernel,
					getDefaultConvolutionPolicy(in_view, out_view));
			cudaDeviceSynchronize();
		}

	};

	auto separable_normal2D = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 2> in(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> out(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> tmp(sizeX2D, sizeY2D);

		auto cb = checkerboard(10, 2, Int2(sizeX/10, sizeY/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("Init");
			copy(cb, view(in));
		}
		{
			auto conv = timer.start("convolution");
			separableConvolution(view(in), view(out), view(tmp), getSeparableGaussian<2>(std_dev));
			cudaDeviceSynchronize();
		}

	};

	auto separable_shared2D = [](Timer & timer){
		auto all = timer.start("All");
		UnifiedImage<float, 2> in(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> out(sizeX2D, sizeY2D);
		UnifiedImage<float, 2> tmp(sizeX2D, sizeY2D);

		auto cb = checkerboard(10, 2, Int2(sizeX/10, sizeY/10), Int2(sizeX2D, sizeY2D));

		{
			auto init = timer.start("Init");
			copy(cb, view(in));
		}
		{
			auto conv = timer.start("convolution");
			separableConvolution(view(in),
					view(out),
					view(tmp),
					getSeparableGaussian<2>(std_dev),
					getDefaultConvolutionPolicy(view(in), view(out)));
			cudaDeviceSynchronize();
		}

	};

	BenchmarkManager bm;
	bm.add("Normal", normal);
	bm.add("Shared memory", shared_memory);
	bm.add("Normal 3D", normal3D);
	bm.add("Shared memory 3D", shared_memory3D);
	bm.add("Separable 2D", separable_normal2D);
	bm.add("Separable Shared 2D", separable_shared2D);

	bm.runAll();
	bm.printAll();

	return 0;
}
