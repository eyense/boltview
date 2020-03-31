
#include <string>
#include <iostream>

#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/unified_image.h>
#include <boltview/create_view.h>
#include <boltview/image_io.h>
#include <boltview/loop_utils.h>
#include <boltview/reduce.h>
#include <boltview/for_each.h>
#include <boltview/subview.h>
#include <boltview/geometrical_transformation.h>
#include <boltview/interpolated_view.h>
#include "io.h"

using namespace bolt;

template<typename TType, int tDim>
using Image = UnifiedImage<TType, tDim>;
// using Image = HostImage<TType, tDim>;

int main(int argc, char** argv) {
	try {
		std::string prefix = "../../data/intestine";
		Image<uint16_t, 3> input_image(Int3(512, 512, 548));
		// std::string prefix = "../../data/lebka2";
		// HostImage<uint16_t, 3> hinput_image(Int3(512, 512, 61));
		load(view(input_image), prefix);

		// UnifiedImage<uint16_t, 3> input_image(hinput_image.size());
		// copy(constView(hinput_image), view(input_image));

		auto inview = constView(input_image);
		auto min_max = minMax(constView(input_image));
		// DeviceImage<uint16_t, 3> scaled_image(Int3(512, 512, 512));
		// scale(makeInterpolatedView<CubicRepeat>(constView(input_image)), view(scaled_image), Float3(1.0f, 1.0f, 8.0f));
                //
                //
		// // auto inview = constView(scaled_image);
		// auto inview = rotatedView(
		// 	constView(scaled_image),
		// 	rotation2VectorsToQuaternion(Float3(0,1,0), Float3(1,1,1)),
		// 	0.5f * scaled_image.size(),
		// 	scaled_image.size());

		auto rotated_view = rotatedView(
			constView(inview),
			rotation2VectorsToQuaternion(Float3(0,1,0), Float3(1,1,0.5f)),
			0.5f * inview.size(),
			inview.size() + Int3(30, 30, 30));
		using MipImage = Image<uint16_t, 2>;
		std::array<MipImage, 4> mips {
			MipImage{ swizzle<0, 2>(inview.size()) },
			MipImage{ swizzle<0, 1>(inview.size()) },
			MipImage{ swizzle<1, 2>(inview.size()) },
			MipImage{ swizzle<0, 2>(rotated_view.size()) } };
		dimensionReduce(
			inview,
			view(mips[0]),
			DimensionValue<1>{},
			uint16_t(0),
			MaxFunctor{});
		std::cout << "-----------------------\n";
		dimensionReduce(
			inview,
			view(mips[1]),
			DimensionValue<2>{},
			uint16_t(0),
			MaxFunctor{});
		std::cout << "-----------------------\n";
		dimensionReduce(
			inview,
			view(mips[2]),
			DimensionValue<0>{},
			uint16_t(0),
			MaxFunctor{});
		std::cout << "-----------------------\n";


		dimensionReduce(
			rotated_view,
			view(mips[3]),
			DimensionValue<1>{},
			uint16_t(0),
			MaxFunctor{});
		auto factor = 255.0f / (min_max[1] - min_max[0]);
		for (int i = 0; i < mips.size(); ++i) {
			dump(constView(mips[i]), "mip" + std::to_string(i));
			Image<uint8_t, 2> tmp(mips[i].size());
			copy((constView(mips[i]) - min_max[0]) * factor, view(tmp));
			saveImage("mip_" + std::to_string(i) + ".jpg", constView(tmp));
		}

		// forEach(view(input_image), [](auto &v){
		// 		uint16_t v2 = ((v & 0xFF) << 8) | ((v & 0xFF00) >> 8);
		// 		v = v2;
		// 	});

		// dump(constView(mip), "mip");
		// dump(subview(constView(input_image), Int3(0, 220, 0), Int3(512, 20, 548)), "center");
		// dump(mirror(constView(input_image), Bool3(false, false, true)), "all");
	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;

	}

	return 0;

}
