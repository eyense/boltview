// Copyright 2016 Eyen SE
// Author: Lukas Marsalek lukas.marsalek@eyen.eu

#pragma once

#include <memory>

#include <boost/gil/typedefs.hpp>
#include <boost/gil/image.hpp>

#include <boltview/cuda_utils.h>
#include <boltview/host_image_view.h>
#include <boltview/copy.h>

namespace bolt {

namespace bg = boost::gil;

///Selects the type to be used as Element type based on PixelElement type of the Gil View.
///It selects either boost::gil::element_type<>::type for unlimited integral pixel types or ... type::base_channel_t for range-limited float pixel type
template<typename TGilViewPixelElementType, typename TElementBaseTypeCheck = void>
struct PixelElementBaseTypeSelector {
	using type = TGilViewPixelElementType;
};

///Specialization for scoped floating point pixel types, which define it as base_channel_t
template<typename TGilViewPixelElementType>
struct PixelElementBaseTypeSelector<TGilViewPixelElementType, typename std::enable_if<std::is_floating_point<typename TGilViewPixelElementType::base_channel_t>::value>::type> {
		using type = typename TGilViewPixelElementType::base_channel_t;
};

///Meta information about the GilViews
template <typename TGilView>
struct GilViewTraits {
	using Pixel_type = typename TGilView::value_type;
	using PixelElement_type = typename boost::gil::element_type<Pixel_type>::type;
	using PixelElementBase_type = typename PixelElementBaseTypeSelector<PixelElement_type>::type;
};

///Template specialization for GilView type to ImageView type mapping
namespace detail {

///To determine the GilView type, we need only the element from the bolt image
///In the default template float is used to cause compile error
template<typename TElement>
struct GilViewTypeFromImageViewTypeImpl {
	using type = float;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<float> {
	using type = boost::gil::gray32f_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<int32_t> {
	using type = boost::gil::gray32s_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<uint32_t> {
	using type = boost::gil::gray32_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<int16_t> {
	using type = boost::gil::gray16s_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<uint16_t> {
	using type = boost::gil::gray16_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<int8_t> {
	using type = boost::gil::gray8s_view_t;
};

template<>
struct GilViewTypeFromImageViewTypeImpl<uint8_t> {
	using type = boost::gil::gray8_view_t;
};

/*
 * This comparator is used to enable determining of min/max of a GIL view by using std::minmax() algorithm.
 * GIL Views return pixel iterators, which however on their own have only operator=, not operator<. This is reasonable in general, since for multi-channel images
 * the operator< does not make always sense (like sorting CMYK pixels) but is well-defined for single-value pixels (a.k.a gray pixels).
 */
struct GrayPixelLessThenComparator {
	template<typename TGrayPixelTypeA, typename TGrayPixelTypeB>
	bool operator() (const TGrayPixelTypeA& item_a, const TGrayPixelTypeB& item_b){
		/* This requirement cannot be enforced by template specialization, since there is no generic "gray pixel" type.
		 * There is only a generic "gray_t" color space but there is no way how to get a pixel type from color space, since this is ambiguous 1:N mapping */
		static_assert(std::is_same<typename bg::color_space_type<TGrayPixelTypeA>::type, bg::gray_t>::value, "GrayPixelLessThenComparator operates only on gray_t color spaces");
		static_assert(std::is_same<typename bg::color_space_type<TGrayPixelTypeB>::type, bg::gray_t>::value, "GrayPixelLessThenComparator operates only on gray_t color spaces");

		return bg::get_color(item_a, bg::gray_color_t()) < bg::get_color(item_b, bg::gray_color_t());
	}
};
};// namespace detail

///Meta constructor of GilView type from ImageView type
template <typename TImageView>
struct GilViewTypeFromImageViewType {
	using type = typename detail::GilViewTypeFromImageViewTypeImpl<typename TImageView::Element>::type;
};

///Image that bridges boost::gil views and bolt host image views. It owns a memory buffer to which it provides both boost::gil view and bolt view.
/// This enables, for example to read into this image through boost::gil I/O operations and then pass the bolt view to device copy operations.
///Or vice-versa, one can create data on GPU through the use of bolt views and transforms and save them using boost::gil I/O.
template<typename TGilView>
class GilAdaptorImage {
public:
	static const bool kIsMemoryBased = true;
	static const int kDimension = 2;

	using GilViewType = TGilView;
	using Element = typename GilViewTraits<GilViewType>::PixelElementBase_type;
	using Policy = DefaultViewPolicy;
	using TIndex = typename Policy::IndexType;
	using HostImageViewType = HostImageView<Element, kDimension, Policy>;
	using HostImageConstViewType = HostImageConstView<Element, kDimension, Policy>;
	using AccessType = Element;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;

	explicit GilAdaptorImage(const SizeType size) :
		size_(size)
	{
		data_.resize(size_[0] * size_[1]);
		strides_ = stridesFromSize(size_);
	}
	explicit GilAdaptorImage(const typename SizeType::Element& width, const typename SizeType::Element& height) :
		GilAdaptorImage(SizeType(width, height))
	{}

	SizeType size(){
		return size_;
	}

	HostImageViewType hostImageView(){
		return HostImageViewType(data_.data(), size_, strides_);
	}

	HostImageConstViewType hostImageConstView(){
		return HostImageConstViewType(data_.data(), size_, strides_);
	}

	GilViewType gilView(){
		return boost::gil::interleaved_view(size_[0], size_[1], reinterpret_cast<typename GilViewTraits<GilViewType>::Pixel_type*>(data_.data()), size_[0] * sizeof(Element));
	}

protected:
	std::vector<Element> data_;
	SizeType strides_;
	SizeType size_;

};

/**
 * Converts between various instances of single-value channels (a.k.a "gray channels") by converting true source range into the data type destination range.
 * This means that the converter first scans the source view to find the true min/max values, which are then converted to the full range of the output data type.
 * This is unlike standard GIL converters, which convert between data type ranges. This is useful for example when dealing with float as a source, since GIL assumes
 * float gray channels come from <0,1> range. This would require additional normalization on the computing site, which is not always desired.
 */
template<typename TPixelTypeDst, typename TGilView>
class MaxRangeConverter {
public:
	using SrcPixel_type = typename TGilView::value_type;
	using DstPixel_type = TPixelTypeDst;
	using DstColorSpace = typename bg::color_space_type<DstPixel_type>::type;
	using SrcColorSpace = typename bg::color_space_type<SrcPixel_type>::type;

	static_assert(std::is_same<SrcColorSpace, DstColorSpace>::value, "MaxRangeConverter operates only on identical color spaces");
	/* This requirement cannot be enforced by template specialization, since there is no generic "gray pixel view" type.
	 * There is only a generic "gray_t" color space but there is no way how to get a view type from color space, since this is ambiguous 1:N mapping */
	static_assert(std::is_same<SrcColorSpace, bg::gray_t>::value, "MaxRangeConverter operates only on gray_t color spaces");

	explicit MaxRangeConverter(const TGilView& source_view) {
		auto src_min_max = std::minmax_element(source_view.begin(), source_view.end(), detail::GrayPixelLessThenComparator());
		srcMin_ = bg::get_color(*src_min_max.first, bg::gray_color_t());
		SrcElement_type src_max = bg::get_color(*src_min_max.second, bg::gray_color_t());
		auto real_src_range = src_max - srcMin_;
		auto dst_range = bg::channel_traits<DstElement_type>::max_value() - bg::channel_traits<DstElement_type>::min_value();
		scaleFactor_ = (real_src_range != 0) ? static_cast<float>(dst_range / real_src_range) : 0.f;
	}

	void operator()(const SrcPixel_type& src, DstPixel_type& dst) const {
		bg::get_color(dst, bg::gray_color_t()) =
				static_cast<DstElement_type>((bg::get_color(src, bg::gray_color_t()) - srcMin_) * scaleFactor_ + bg::channel_traits<DstElement_type>::min_value());
	}

private:
	using SrcElement_type = typename bg::element_type<SrcPixel_type>::type;
	using DstElement_type = typename bg::element_type<DstPixel_type>::type;
	SrcElement_type srcMin_;
	float scaleFactor_;
};

} // namespace bolt
