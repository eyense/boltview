#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/copy.h>
#include <boltview/create_view.h>
#include <boltview/for_each.h>
#include <boltview/math/vector.h>
#include "io.h"

using namespace bolt;


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


template<typename TOutputView>
void generateImage(TOutputView output_view, bolt::Region<2, float> domain) {
	forEachPosition(output_view, Mandelbrot{output_view.size(), domain});
}


void runOnDevice(bolt::Int2 size, bolt::Region<2, float> domain) {
	DeviceImage<Vector<uint8_t, 3>, 2> device_image(size);
	generateImage(view(device_image), domain);

	HostImage<Vector<uint8_t, 3>, 2> output_image(size);
	copy(constView(device_image), view(output_image));
	saveImage("device_output.jpg", constView(output_image));
}

void runOnHost(bolt::Int2 size, bolt::Region<2, float> domain) {
	HostImage<Vector<uint8_t, 3>, 2> output_image(size);
	generateImage(view(output_image), domain);

	saveImage("host_output.jpg", constView(output_image));
}
