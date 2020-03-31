// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <cstring>
#include <memory>

#include <boltview/cuda_utils.h>
#include <boltview/host_image_view.h>

namespace bolt {

/// \addtogroup Images
/// @{

/// Device image representation, which owns the data.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class HostImage {
public:
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Element = TElement;
	using ViewType = HostImageView<TElement, tDimension, Policy>;
	using ConstViewType = HostImageConstView<const TElement, tDimension, Policy>;
	static const int kDimension = tDimension;

	HostImage() = default;

	explicit HostImage(SizeType size)
	{
		reallocate(size);
	}

	template<typename TValue, typename TDeleter>
	HostImage(SizeType size, SizeType strides, std::unique_ptr<TValue, TDeleter> &&ptr) :
		size_(size),
		strides_(strides),
		host_ptr_(std::move(ptr))
	{
	}

	explicit HostImage(int size)
	{
		static_assert(tDimension == 1, "Only for dimension 1");
		reallocate(SizeType(size));
	}

	HostImage(TIndex width, TIndex height)
	{
		static_assert(tDimension == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
		reallocate(SizeType(width, height));
	}

	HostImage(TIndex width, TIndex height, TIndex depth)
	{
		static_assert(tDimension == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
		reallocate(SizeType(width, height, depth));
	}

	HostImage(const HostImage &other) = delete;
	HostImage &operator=(const HostImage &other) = delete;

	HostImage(HostImage &&other) = default;
	HostImage &operator=(HostImage &&other) = default;

	~HostImage() = default;

	/// \return Image size in each dimension.
	SizeType size() const {
		return size_;
	}

	/// \return Offsets needed to increase element coordinate by 1 for each coordinate axis.
	StridesType strides() const {
		return strides_;
	}

	TElement *pointer() const {
		return host_ptr_.get();
	}

	/// Create view for whole image, which can be used for modification of image data.
	ViewType view() {
		return ViewType(host_ptr_.get(), size_, strides_);
	}

	/// Create view for whole image, which can be used for const access to the image data.
	ConstViewType constView() const {
		return ConstViewType(host_ptr_.get(), size_, strides_);
	}

	void clear() {
		std::memset(host_ptr_.get(), 0, sizeof(TElement) * product(Vector<uint64_t, SizeType::kDimension>(size_)));
	}

protected:
	void reallocate(SizeType size) {
		uint64_t num_elements = product(Vector<uint64_t, SizeType::kDimension>(size));
		host_ptr_ = std::unique_ptr<TElement []>(new TElement[num_elements]);
		size_ = size;
		strides_ = stridesFromSize(size_);
	}

	SizeType size_;
	StridesType strides_;
	std::unique_ptr<TElement [], std::function<void(TElement*)>> host_ptr_;
};

/// @}

}  // namespace bolt
