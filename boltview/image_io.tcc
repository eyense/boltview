// Copyright 2017 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <fstream>
#include <sstream>
#include <string>

#include <boltview/copy.h>
#include <boltview/cuda_defines.h>
#include <boltview/host_image.h>
#include <boltview/math/vector.h>
#include <boltview/math/complex.h>

#ifdef __CUDACC__
// #include <boltview/fft_utils.h>
#include <boltview/device_image.h>
#endif

namespace bolt {

namespace detail {

//TODO(johny) - replace these overloads by type trait manipulations
inline char getIdentifier(uint8_t  /*arg*/) {
	return 'U';
}

inline char getIdentifier(int8_t  /*arg*/) {
	return 'S';
}

inline char getIdentifier(uint16_t  /*arg*/) {
	return 'U';
}

inline char getIdentifier(int16_t  /*arg*/) {
	return 'S';
}

inline char getIdentifier(int  /*arg*/) {
	return 'S';
}

inline char getIdentifier(float  /*arg*/) {
	return 'F';
}

inline char getIdentifier(unsigned int  /*arg*/) {
	return 'U';
}

inline char getIdentifier(double  /*arg*/) {
	return 'D';
}

#ifdef __CUDACC__
inline char getIdentifier(cufftComplex arg) {
	return 'C';
}

inline char getIdentifier(cufftDoubleComplex arg) {
	return 'Z';
}
#endif

inline char getIdentifier(HostComplexType  /*arg*/) {
	return 'Z';
}

template<class TElement>
std::string getTypeSuffix() {
	return std::to_string(sizeof(TElement) * 8) + getIdentifier(TElement());
}

inline std::string getChannelSuffix(int channels) {
	return (channels == 1) ? "" : 'C' + std::to_string(channels);
}

template<class TT>
struct SuffixHelper {
	using Type = TT;
	static const int kSize = 1;
};

template<class TElement, int tDimension>
struct SuffixHelper<Vector<TElement, tDimension>> {
	using Type = TElement;
	static const int kSize = tDimension;
};

template<typename TView>
std::string getRawFilename(typename TView::SizeType size, std::string prefix) {
	using Helper = SuffixHelper<typename TView::Element>;
	std::stringstream buffer;
	buffer << prefix;
	buffer << '_' << getTypeSuffix<typename Helper::Type>();
	buffer << getChannelSuffix(Helper::kSize) << '_';
	buffer << size[0];
	for (int i = 1; i < TView::SizeType::kDimension; i++) {
		buffer << 'x' << size[i];
	}
	buffer << ".raw";
	return buffer.str();
}

template<typename TView, typename std::enable_if<TView::kDimension == 1>::type * = nullptr>
std::string getRawFilename(int size, std::string prefix) {
	using Helper = SuffixHelper<typename TView::Element>;
	std::stringstream buffer;
	buffer << prefix;
	buffer << '_' << getTypeSuffix<typename Helper::Type>();
	buffer << getChannelSuffix(Helper::kSize) << '_';
	buffer << size;
	buffer << "x1.raw";
	return buffer.str();
}

template<bool tHostImage, bool tMemoryBased>
struct SafeCopyHelper {
	template <typename TFromView, typename TToView>
	static void safeCopy(TFromView from_view, TToView to_view) {
		copy(from_view, to_view);
	}
};

#ifdef __CUDACC__
template<>
struct SafeCopyHelper<false, false> {
	template <typename TFromView, typename TToView>
	static void safeCopy(TFromView from_view, TToView to_view) {
		using ImageType = DeviceImage<typename TFromView::Element, TFromView::kDimension>;
		ImageType tmp(from_view.size());
		copy(from_view, tmp.view());
		copy(tmp.view(), to_view);
	}
};
#endif

template<typename TView>
class StorageHelper {
public:
	using HostImageType = HostImage<typename TView::Element, TView::kDimension>;
	using HostImageViewType = decltype(std::declval<HostImageType>().view());

	explicit StorageHelper(typename TView::SizeType size) : image_(size) {}

	explicit StorageHelper(int size) : image_(Int1(size)) {
		static_assert(TView::kDimension == 1, "Only for 1D");
	}

	HostImageViewType view() {
		return image_.view();
	}

	int64_t safeElementCount() const {
		Vector<int64_t, TView::SizeType::kDimension> size(image_.size());
		return product(size);
	}

protected:
	HostImageType image_;
};

template<typename TView>
class DumpStorageHelper : public StorageHelper<TView> {
public:
	explicit DumpStorageHelper(TView view) : StorageHelper<TView>(view.size()) {
		SafeCopyHelper<TView::kIsHostView, IsMemcpyAble<TView>::value>::safeCopy(view, this->view());
	}
};

template<typename TImage>
class LoadStorageHelper : public StorageHelper<TImage> {
public:
	explicit LoadStorageHelper(typename TImage::SizeType size) : StorageHelper<TImage>(size) {}

	TImage getImage() {
		TImage tmp(this->image_.size());
		copy(this->image_.constView(), tmp.view());
		return tmp;
	}
};

template<typename TImage>
LoadStorageHelper<TImage> loadImpl(typename TImage::SizeType size, std::string prefix) {
	LoadStorageHelper<TImage> tmp(size);
	std::string filename = getRawFilename<TImage>(size, prefix);

	BOLT_DFORMAT("Loading view from file: %1%", filename);
	std::ifstream in;
	in.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	in.open(filename, std::ifstream::in | std::ifstream::binary);

	// TODO(tom): check file size before reading
	int64_t num_elements = tmp.safeElementCount();
	for (int64_t i = 0; i < num_elements; ++i) {
		auto &element = linearAccess(tmp.view(), i);
		in.read(reinterpret_cast<char *>(&element), sizeof(element));
	}
	return tmp;
}

}  // namespace detail

template<typename TView>
std::string dump(TView view, std::string prefix) {
	detail::DumpStorageHelper<TView> tmp(view);
	std::string filename = detail::getRawFilename<TView>(view.size(), prefix);

	BOLT_DFORMAT("Dumping view to file: %1%", filename);
	std::ofstream out;
	out.exceptions(std::ofstream::failbit | std::ofstream::badbit);
	out.open(filename, std::ofstream::out | std::ofstream::binary);

	int64_t num_elements = tmp.safeElementCount();
	for (int64_t i = 0; i < num_elements; ++i) {
		auto element = linearAccess(tmp.view(), i);
		out.write(reinterpret_cast<const char *>(&element), sizeof(element));
	}
	return filename;
}

template<typename TImage>
TImage load(typename TImage::SizeType size, std::string prefix) {
	auto tmp_storage = detail::loadImpl<TImage>(size, prefix);
	return tmp_storage.getImage();
}

template<typename TView>
void load(TView view, std::string prefix) {
	auto tmp_storage = detail::loadImpl<TView>(view.size(), prefix);
	detail::SafeCopyHelper<TView::kIsHostView, IsMemcpyAble<TView>::value>
		::safeCopy(tmp_storage.view(), view);
}

}  // namespace bolt
