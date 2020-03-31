// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomaskrupka@eyen.se

#pragma once

#if !defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)

#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/device_image_view_base.h>
#include <boltview/view_traits.h>
#include <boltview/subview.h>
#include <boltview/texture_image_types.h>

#include <algorithm>
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
#error "Texture object usage requires at least Kepler functionality"
#endif

namespace bolt {

template<typename TCudaType, typename TReturnValue>
BOLT_DECL_DEVICE
TReturnValue accessTexture(const cudaTextureObject_t &tex_object, float index) {
	return TCudaType::toVec(tex1D<TCudaType::type>(
			tex_object,
			index + TCudaType::offset()));
}

template<typename TCudaType, typename TReturnValue>
BOLT_DECL_DEVICE
TReturnValue accessTexture(const cudaTextureObject_t &tex_object, Float2 index) {
	return TCudaType::toVec(tex2D<TCudaType::type>(
			tex_object,
			index[0] + TCudaType::offset(),
			index[1] + TCudaType::offset()));
}

template<typename TCudaType, typename TReturnValue>
BOLT_DECL_DEVICE
TReturnValue accessTexture(const cudaTextureObject_t &tex_object, Float3 index) {
	return TCudaType::toVec(tex3D<TCudaType::type>(
			tex_object,
			index[0] + TCudaType::offset(),
			index[1] + TCudaType::offset(),
			index[2] + TCudaType::offset()));
}

/// \addtogroup Views
/// @{

/// View to the whole texture image, which owns the data.
/// It provides only constant access to the data. It is usable on both host/device sides.
/// Data is accessed via Float coordinates and hw interpolated values are returned
template<typename TElement, int tDimension, typename TCudaType = DefaultCudaType<TElement>>
class TextureImageConstView : public DeviceImageViewBase<tDimension> {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = DefaultViewPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using CoordinateType = Vector<float, tDimension>;
	using Predecessor = DeviceImageViewBase<tDimension>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	TextureImageConstView(cudaArray *cuda_array, cudaTextureObject_t tex_object, SizeType size, SizeType strides) :
		Predecessor(size),
		strides_(strides),
		cuda_array_(cuda_array),
		tex_object_(tex_object)
	{
		//D_FORMAT("Texture image const view:\n\tsize: %1%\n\tstrides: %2%", size, strides);
	}

	BOLT_DECL_HYBRID
	TextureImageConstView() :
		Predecessor(SizeType()),
		cuda_array_(nullptr)
	{}

	BOLT_DECL_HYBRID
	TextureImageConstView(const TextureImageConstView &view, IndexType corner, SizeType size) :
		Predecessor(size),
		strides_(view.strides_),
		cuda_array_(view.cuda_array_),
		tex_object_(view.tex_object_),
		corner_(corner)
	{
		#ifndef __CUDA_ARCH__
		BOLT_DFORMAT("TextureImageConstView %1% - %2%", corner, size);
		#endif

	}

	BOLT_DECL_DEVICE
	TElement operator[](IndexType index) const {
		BOLT_ASSERT(cuda_array_ != nullptr);
		return access(static_cast<CoordinateType>(index));
	}

	BOLT_DECL_DEVICE
	TElement access(CoordinateType coordinates) const {
		return accessTexture<TCudaType, TElement>(tex_object_, corner_ + coordinates);
	}

	BOLT_DECL_HYBRID
	SizeType strides() const {
		return strides_;
	}

	BOLT_DECL_HYBRID
	const cudaArray *array() const {
		return cuda_array_;
	}

	BOLT_DECL_HYBRID
	cudaTextureObject_t textureObject() const {
		return tex_object_;
	}

	BOLT_DECL_HYBRID
	TextureImageConstView<TElement, tDimension, TCudaType> subview(const IndexType &corner, const SizeType &size) const {
		#ifndef __CUDA_ARCH__
		BOLT_DFORMAT("TextureImageConstView::subview %1% - %2%", corner, size);
		#endif

		return TextureImageConstView<TElement, tDimension, TCudaType>(*this, corner, size);
	}

	template<int tSliceDimension>
	BOLT_DECL_HYBRID
	SliceImageView<TextureImageConstView, tSliceDimension, kIsDeviceView> slice(typename IndexType::Element slice) const {
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->Size()[tSliceDimension]);
		return SliceImageView<TextureImageConstView, tSliceDimension, kIsDeviceView>(*this, slice);
	}

	IndexType corner() const {
		return corner_;
	}
protected:
	SizeType strides_;
	cudaArray *cuda_array_;
	cudaTextureObject_t tex_object_;
	IndexType corner_;
};


template<typename TElement, int tDimension, typename TCudaType>
struct IsTextureView<TextureImageConstView<TElement, tDimension, TCudaType>>  : std::integral_constant<bool, true> {};

/// View to the whole texture image, which owns the data.
/// It provides reference access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TCudaType = DefaultCudaType<TElement>>
class TextureImageView : public TextureImageConstView<TElement, tDimension> {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = DefaultViewPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using CoordinateType = Vector<float, tDimension>;
	using Predecessor = TextureImageConstView<TElement, tDimension>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	TextureImageView(cudaArray *cuArray, cudaTextureObject_t texObject, SizeType size, SizeType strides) :
		Predecessor(cuArray, texObject, size, strides)
	{}

	BOLT_DECL_HYBRID
	TextureImageView() {}

	BOLT_DECL_HYBRID
	TextureImageView(const TextureImageView &view, IndexType corner, SizeType size) :
		Predecessor(
			static_cast<const Predecessor &>(view),
			corner,
			size)
	{}

	/// \return Pointer for raw access to the data buffer.
	BOLT_DECL_HYBRID
	cudaArray *array() const {
		return this->cuda_array_;
	}

	BOLT_DECL_HYBRID
	TextureImageView<TElement, tDimension, TCudaType> subview(const IndexType &corner, const SizeType &size) const {
		#ifndef __CUDA_ARCH__
		BOLT_DFORMAT("TextureImageView::subview %1% - %2%", corner, size);
		#endif
		return TextureImageView<TElement, tDimension, TCudaType>(*this, corner, size);
	}

	template<int tSliceDimension>
	BOLT_DECL_HYBRID
	SliceImageView<TextureImageView, tSliceDimension, kIsDeviceView> slice(typename IndexType::Element slice) const {
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->Size()[tSliceDimension]);
		return SliceImageView<TextureImageView, tSliceDimension, kIsDeviceView>(*this, slice);
	}
};

template<typename TElement, int tDimension, typename TCudaType>
struct IsTextureView<TextureImageView<TElement, tDimension, TCudaType>>  : std::integral_constant<bool, true> {};

template<typename TElement, int tDimension, typename TCudaType>
struct IsInterpolatedView<TextureImageView<TElement, tDimension, TCudaType>> : std::integral_constant<bool, true> {};


}  // namespace bolt

#include "texture_image_view.tcc"
