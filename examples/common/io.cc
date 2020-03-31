#include "io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <boltview/host_image.h>
#include <boltview/host_image_view.h>
#include <boltview/exceptions.h>

#include <boost/scope_exit.hpp>

namespace bolt {

ImageVariant
loadImage(boost::filesystem::path filename) {

	auto deleter = [](auto data) { stbi_image_free(data); };
	int x,y,n;

	unsigned char *data = stbi_load(filename.c_str(), &x, &y, &n, 0);
	if (data == nullptr) {
		BOLT_THROW(ImageFileReadError());
	}

	switch (n) {
	case 1: {
			std::unique_ptr<unsigned char [], decltype(deleter)> ptr(data, deleter);
			return HostImage<uint8_t, 2>(Int2(x, y), Int2(1, x), std::move(ptr));
		}
	case 3: {
			std::unique_ptr<Vector<uint8_t, 3> [], decltype(deleter)> ptr(reinterpret_cast<Vector<uint8_t, 3> *>(data), deleter);
			return HostImage<Vector<uint8_t, 3>, 2>(Int2(x, y), Int2(1, x), std::move(ptr));
		}
	default :
		BOLT_THROW(UnsupportedFileFormat());
	}
	return {};
}

}  // namespace bolt
