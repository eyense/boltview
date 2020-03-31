#pragma once

#include "stb_image.h"
#include "stb_image_write.h"

#include <boltview/host_image.h>
#include <boltview/host_image_view.h>
#include <boltview/exceptions.h>

#include <boost/variant.hpp>
#include <boost/algorithm/string.hpp>

namespace bolt {
using ImageVariant = boost::variant<
	bolt::HostImage<uint8_t, 2>,
	bolt::HostImage<Vector<uint8_t, 3>, 2>,
	bolt::HostImage<Vector<uint8_t, 4>, 2>,
	bolt::HostImage<float, 2>,
	bolt::HostImage<Vector<float, 3>, 2>,
	bolt::HostImage<Vector<float, 4>, 2>>;

struct UnsupportedFileFormat: BoltError {};

struct ImageFileWriteError: BoltError {};
struct ImageFileReadError: BoltError {};

ImageVariant
loadImage(boost::filesystem::path);


using FileExtensionErrorInfo = boost::error_info<struct tag_extension, std::string>;

template<bool tIsScalar, typename TType>
struct ChannelCount{
	static constexpr int value = 1;
};

template<typename TType>
struct ChannelCount<false, TType>{
	static constexpr int value = TType::kDimension;
};

template<typename TImageView>
void saveImage(boost::filesystem::path filename, TImageView output_view) {
	static constexpr int kQuality = 90;
	auto ext = boost::algorithm::to_lower_copy(filename.extension().string());

	/*TODO*/
	int channel_count = ChannelCount<std::is_scalar<typename TImageView::Element>::value, typename TImageView::Element>::value;
	if (ext == ".jpg") {
		if (0 == stbi_write_jpg(filename.c_str(), output_view.size()[0], output_view.size()[1], channel_count, output_view.pointer(), kQuality)) {
			BOLT_THROW(ImageFileWriteError() << FileExtensionErrorInfo(ext));
		}
	} else if (ext == ".png") {
		 if (0 == stbi_write_png(filename.c_str(), output_view.size()[0], output_view.size()[1], channel_count, output_view.pointer(), output_view.size()[0]*channel_count)) {
			BOLT_THROW(ImageFileWriteError() << FileExtensionErrorInfo(ext));
		 }
	} else {
		BOLT_THROW(UnsupportedFileFormat() << FileExtensionErrorInfo(ext));
	}
}

}  // namespace bolt
