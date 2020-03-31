
#include <string>
#include <iostream>

#include <boltview/host_image.h>
#include <boltview/create_view.h>
#include <boltview/image_io.h>
#include <boltview/loop_utils.h>
#include <boltview/for_each.h>
#include <boltview/math/vector.h>
#include "io.h"


using namespace bolt;

// struct ComplexPlaneMapper

struct Mandelbrot
{
	static constexpr int max_iteration = 500;

	Mandelbrot(
		Int2 aExtents,
		Region<2, float> r = { Float2(-2.5f, -1.0f), Float2(3.5f, 2.0f) }):
			extents(aExtents),
			region(r)
	{}

	static BOLT_DECL_HYBRID
	Vector<uint8_t, 3> colorMap(int iteration) {
		Vector<uint8_t, 3> tmp;
		tmp[0] = iteration % 256;
		tmp[1] = (iteration * 7) % 256;
		tmp[2] = (iteration * 13) % 256;
		return tmp;
	}

	template<typename TValue>
	BOLT_DECL_HYBRID void
	operator()(TValue &val, Int2 position) const {
		auto coords = product(div(Float2(position), extents), region.size) + region.corner;

		HostComplexType z0{ coords[0], coords[1] };
		HostComplexType z{0, 0};

		int iteration = 0;

		while (magSquared(z) < 4  &&  (iteration < max_iteration)) {
			z = z*z  + z0;
			++iteration;
		}
		val = colorMap(iteration);
	}
	Int2 extents;
	Region<2, float> region;
};



int main(int argc, char** argv) {
	try {
		HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(4282, 2848));
		// HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(30000, 20000));
		DeviceImage<Vector<uint8_t, 3>, 2> device_image(output_image.size());

		forEachPosition(view(device_image), Mandelbrot{device_image.size()/*, { Float2(-1.03f, 0.29f), Float2(0.06f, 0.045f) }*/});

		copy(constView(device_image), view(output_image));
		saveImage("output.jpg", constView(output_image));
	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;
	}

	return 0;

}
