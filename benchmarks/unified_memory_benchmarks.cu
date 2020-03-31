// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.eu

/// Test UnifiedMemory

#include <unistd.h>


#include <boltview/device_image.h>
#include <boltview/host_image.h>
#include <boltview/unified_image.h>
#include <boltview/subview.h>
#include <boltview/procedural_views.h>
#include <boltview/create_view.h>
#include <boltview/transform.h>
#include <boltview/copy.h>

#include <fstream>
#include <iostream>

#include "benchmarks.h"

using namespace bolt;
using namespace std;


void dumpMemory(const char * name, float * ptr, const int sizex, const int sizey){
	std::cout << "Writing memory dump: " << name << '\n';
	ofstream ofs;
	ofs.open(name, ios::out | ios::binary);
	ofs.write((const char*)ptr, sizex*sizey*sizeof(*ptr));
	ofs.close();
}


int main() {
	BenchmarkManager bm;
	const int SIZE = 41000;

	// CheckerBoard to HostImage,  copy to device, transform and copy back to device
	auto device = [](Timer &timer){
		auto all = timer.start("all");
		auto checker_board = checkerboard(2, 1, Int2(1, 1), Int2(SIZE, SIZE));
		HostImage<float, 2> host_image(SIZE, SIZE);
		DeviceImage<float, 2> device_image(SIZE/2, SIZE/2);
		{
			auto copy_interval = timer.start("checkerboard copy");
			copy(checker_board, view(host_image));
		}
		auto copy_transform = timer.start("copy, transform, copy");
		{
			for(int i = 0; i < 2; ++i){
				for (int j = 0; j < 2; ++j) {
					copyAsync(subview(constView(host_image), Int2(SIZE/2*j, SIZE/2*i), Int2(SIZE/2, SIZE/2)), view(device_image));
					transform(constView(device_image), view(device_image), SquareRootFunctor());
					copyAsync(constView(device_image), subview(view(host_image), Int2(SIZE/2*j, SIZE/2*i), Int2(SIZE/2, SIZE/2)));
				}
			}
			cudaDeviceSynchronize();
		}
		// dumpMemory("device_dump", host_image.View().Pointer(), SIZE, SIZE);
	};

	// Copy CheckBoard to UnifiedImage and transform
	auto unified = [](Timer &timer){
		auto all = timer.start("all");
		auto checker_board = checkerboard(2, 1, Int2(1, 1), Int2(SIZE, SIZE));
		UnifiedImage<float, 2> unified_image(SIZE, SIZE);
		{
			auto to_unif = timer.start("checkerboard copy");
			copy(checker_board, view(unified_image));
		}
		{
			auto transform_interval = timer.start("transform");
			transform(constView(unified_image), view(unified_image), SquareRootFunctor());
			cudaDeviceSynchronize();
		}
		// dumpMemory("unified_dump", unified_image.View().Pointer(), SIZE, SIZE);
	};

	// Copy CheckBoard to UnifiedImage and transform with prefetch
	auto unified_prefetch = [](Timer &timer){
		auto all = timer.start("All");
		auto checker_board = checkerboard(2, 1, Int2(1, 1), Int2(SIZE, SIZE));
		UnifiedImage<float, 2> unified_image(SIZE, SIZE);
		{
			auto copy_interval = timer.start("CheckerBoard copy");
			copy(checker_board, view(unified_image));
		}
		{
			auto copy_transform = timer.start("prefetch, transform");
			for(int i = 0; i < 4; ++i) {
				cudaMemPrefetchAsync(
					view(unified_image).pointer() + i * SIZE * SIZE / 4,
					SIZE*SIZE,
					0);
				auto sview = subview(view(unified_image), Int2(0, SIZE/4*i), Int2(SIZE, SIZE/4));
				transform(sview, sview, SquareRootFunctor());
			}
			cudaDeviceSynchronize();
		}
		// dumpMemory("prefetch_dump", unified_image.View().Pointer(), SIZE, SIZE);
	};

	// Copy CheckBoard to UnifiedImage and transform with prefetch (PrefetchView function)
	auto prefetch_function = [](Timer &timer){
		auto all = timer.start("All");
		auto checker_board = checkerboard(2, 1, Int2(1, 1), Int2(SIZE, SIZE));
		UnifiedImage<float, 2> unified_image(SIZE, SIZE);
		{
			auto copy_interval = timer.start("CheckerBoard copy");
			copy(checker_board, view(unified_image));
		}
		{
			auto copy_transform = timer.start("prefetch, transform");
			for(int i = 0; i < 4; ++i) {
				auto sview = subview(view(unified_image), Int2(0, SIZE/4*i), Int2(SIZE, SIZE/4));
				prefetchView(sview, 0);
				transform(sview, sview, SquareRootFunctor());
			}
			cudaDeviceSynchronize();
		}
		// dumpMemory("prefetch_function_dump", unified_image.View().Pointer(), SIZE, SIZE);
	};

	bm.add("DeviceImage", device);
	bm.add("UnifiedImage", unified);
	bm.add("UnifiedImage prefetch", unified_prefetch);
	bm.add("UnifiedImage prefetch2", prefetch_function);

	bm.runAll();
	std::cout << "SIZE: " << SIZE << "^2\n";
	bm.printAll();

	return 0;
}
