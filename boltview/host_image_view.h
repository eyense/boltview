// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <type_traits>

#include <boltview/cuda_utils.h>
#include <boltview/math/vector.h>
#include <boltview/host_image_view_base.h>
#include <boltview/exceptions.h>


namespace bolt {

/// \addtogroup Views
/// @{


/// View to the part or whole device image, which owns the data.
/// It provides only constant access to the data.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class HostImageConstView : public HostImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = false;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HostImageViewBase<tDimension, Policy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;

	// TODO(johny) iterator access for range based for loops

	HostImageConstView(TElement *host_ptr, SizeType size, SizeType strides) :
		Predecessor(size),
		host_ptr_(host_ptr),
		strides_(strides)
	{
		//D_FORMAT("Size %1%, strides %2%", this->size_, this->strides_);
	}

	HostImageConstView() :
		Predecessor(SizeType()),
		host_ptr_(nullptr)
	{}

	/// \return N-dimensional vector of strides - number of elements,
	/// which must be skipped to increase index by one.
	StridesType strides()  const {
		return strides_;
	}

	const TElement *pointer() const {
		return host_ptr_;
	}

	TElement operator[](IndexType index) const {
		return host_ptr_[linearIndex(strides_, index)];
	}

	bool hasContiguousMemory() const {
		return stridesFromSize(this->size()) == strides_;
		// TODO(johny) - support also reordered memory access.
		// return -1 != Find(strides_, 1) || -1 != Find(strides_, -1);
	}

	/// Creates view for part of this view.
	/// \param corner Index of topleft corner
	/// \param size size of the subimage
	HostImageConstView<TElement, tDimension, Policy>
	subview(const IndexType &corner, const SizeType &size) const {
		BOLT_ASSERT(corner >= SizeType());
		BOLT_ASSERT(corner < this->size());
		BOLT_ASSERT((corner + size) <= this->size());
		return HostImageConstView<TElement, tDimension, Policy>(
			this->host_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	HostImageConstView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->host_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return HostImageConstView<TElement, tDimension - 1, Policy>(
				slice_corner,
				RemoveDimension(this->size_, tSliceDimension),
				RemoveDimension(this->strides_, tSliceDimension));
	}

	HostImageConstView<TElement, tDimension, Policy> constView() const {
		return *this;
	}

protected:
	// TODO(johny) - view with offsets
	TElement *host_ptr_;
	StridesType strides_;
};


/// View to the part or whole device image, which owns the data.
/// It provides reference access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class HostImageView : public HostImageConstView<TElement, tDimension, TPolicy> {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = false;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HostImageConstView<TElement, tDimension, Policy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element &;

	// TODO(johny) iterator access for range based for loops

	HostImageView(TElement *host_ptr, SizeType size, SizeType strides) :
		Predecessor(host_ptr, size, strides)
	{}

	HostImageView() = default;

	/// \return Pointer for raw access to the data buffer.
	TElement *pointer() const {
		return this->host_ptr_;
	}

	TElement &operator[](IndexType index) const {
		return this->host_ptr_[linearIndex(this->strides_, index)];
	}

	HostImageView<TElement, tDimension, Policy>
	subview(const IndexType &corner, const SizeType &size) const {
		//D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		return HostImageView<TElement, tDimension, Policy>(
			this->host_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	template<int tSliceDimension>
	HostImageView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->host_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return HostImageView<TElement, tDimension - 1, Policy>(
				slice_corner,
				removeDimension(this->size_, tSliceDimension),
				removeDimension(this->strides_, tSliceDimension));
	}

	operator HostImageConstView<const TElement, tDimension, Policy> () {
		return HostImageConstView<const TElement, tDimension, Policy> (this->pointer(), this->size(), this->strides());
	}

	HostImageConstView<const TElement, tDimension, Policy> constView() const {
		return HostImageConstView<const TElement, tDimension, Policy>(
			this->pointer(), this->size(), this->strides());
	}
};

template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
HostImageConstView<const TElement, tDimension, TPolicy>
makeHostImageConstView(
	const TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	TPolicy /*policy*/ = TPolicy())
{
	return makeHostImageConstView<TElement, tDimension, TPolicy>(buffer, size, stridesFromSize(size));
}

template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
HostImageConstView<const TElement, tDimension, TPolicy>
makeHostImageConstView(
	const TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	Vector<typename TPolicy::IndexType, tDimension> strides,
	TPolicy /*policy*/ = TPolicy())
{
	return HostImageConstView<const TElement, tDimension, TPolicy>(buffer, size, strides);
}

template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
HostImageView<TElement, tDimension, TPolicy>
makeHostImageView(
	TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	TPolicy /*policy*/ = TPolicy())
{
	return makeHostImageView<TElement, tDimension, TPolicy>(buffer, size, stridesFromSize(size));
}

template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
HostImageView<TElement, tDimension, TPolicy>
makeHostImageView(
	TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	Vector<typename TPolicy::IndexType, tDimension> strides,
	TPolicy /*policy*/ = TPolicy())
{
	return HostImageView<TElement, tDimension, TPolicy>(buffer, size, strides);
}

/// @}

}  // namespace bolt
