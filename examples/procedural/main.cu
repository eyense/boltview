
#include <string>
#include <iostream>
#include <utility>

#include <boltview/host_image.h>
#include <boltview/create_view.h>
#include <boltview/image_io.h>
#include <boltview/procedural_views.h>
#include <boltview/for_each.h>
#include <boltview/math/vector.h>
#include <boltview/variadic_templates.h>
#include "io.h"


using namespace bolt;

template<int... tArgs>
auto generate(Int2 size, std::integer_sequence<int, tArgs...> seq) {

	return fold(
		[](auto first, auto second) { return first + second; },
		// (checkerboard(uint8_t(0), uint8_t(255), Int2(power(3, tArgs), FillTag{}), size) * checkerboard(uint8_t(0), uint8_t(255), Int2(power(3, tArgs), power(3, tArgs+1)), size))
		checkerboard(uint8_t(0), uint8_t(255/seq.size()), Int2(power(3, tArgs), FillTag{}), size)

		...);
}



int main(int argc, char** argv) {
	try {
		Int2 size{2187, 2187};

		auto im = generate(size, std::make_integer_sequence<int, 7>{});

		HostImage<uint8_t, 2> output_image(size);
		copy(im, view(output_image));
		saveImage("output.jpg", constView(output_image));
	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;
	}

	return 0;

}
