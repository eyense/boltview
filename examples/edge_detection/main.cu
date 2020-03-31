
#include <string>
#include <iostream>

#include <boltview/host_image.h>
#include <boltview/create_view.h>
#include <boltview/image_io.h>
#include <boltview/loop_utils.h>
#include <boltview/transform.h>
#include "io.h"

using namespace bolt;

static const Float2 kWeights[3][3] = {
	{{-1.0f, -1.0f},{-2.0f, 0.0f}, {-1.0f, 1.0f}},
	{{ 0.0f, -2.0f},{ 0.0f, 0.0f}, { 0.0f, 2.0f}},
	{{ 1.0f, -1.0f},{ 2.0f, 0.0f}, { 1.0f, 1.0f}}
};

struct SobelEdge {
	template<typename TLocator>
	auto operator()(TLocator locator) const {
		auto val = sumEachNeighbor(
				Int2(-1, -1),
				Int2(2, 2),
				Float2{},
				[&locator](Int2 index) {
					return locator[index] * (kWeights[index[0] + 1][index[1] + 1]);
				});
		auto mag = clamp(3*norm(val), 0.0f, 255.0f);
		return mag < 50.0f ? 0.0f : mag;
	}

};


int main(int argc, char** argv) {
	try {
		std::string prefix = "../../data/bee";

		// HostImage<uint8_t, 2> input_image(Int2(4282, 2848));

		auto imageVar = loadImage("../../data/bee_bw.png");
		auto input_image = boost::get<HostImage<uint8_t, 2>>(std::move(imageVar));
		HostImage<float, 2> output_image(input_image.size());

		transformLocator(constView(input_image), view(output_image), SobelEdge{});

		copy(constView(output_image), view(input_image));
		saveImage("edge_detection.jpg", constView(input_image));
	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;
	}

	return 0;

}
