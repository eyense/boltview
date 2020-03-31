// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#if ! defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)


#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <boltview/cuda_defines.h>
#include <boltview/cuda_utils.h>
#include <boltview/device_image_view_base.h>
#include <boltview/exceptions.h>
#include <boltview/functors.h>

// TODO(johny) - restrictive view which will wrap hybrid view as device only or host only

/// Declarations of various image views - memory based, procedural, lazy evaluations.

namespace bolt {

/// \addtogroup Views
/// @{

/// View to the part or whole device image, which owns the data.
/// It provides only constant access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class DeviceImageConstView : public DeviceImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = DeviceImageViewBase<tDimension, TPolicy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	DeviceImageConstView(TElement *device_ptr, SizeType size, SizeType strides) :
		Predecessor(size),
		strides_(strides),
		device_ptr_(device_ptr)
	{
		// D_FORMAT("Device view:\n\tsize: %1%\n\tstrides: %2%", size, strides);
	}

	BOLT_DECL_HYBRID
	DeviceImageConstView() :
		Predecessor(SizeType()),
		device_ptr_(nullptr)
	{}

	/// \return N-dimensional vector of strides - number of elements,
	/// which must be skipped to increase index by one.
	BOLT_DECL_HYBRID
	StridesType strides()  const {
		return strides_;
	}


	BOLT_DECL_HYBRID
	const Element *pointer() const {
		return device_ptr_;
	}

	BOLT_DECL_DEVICE
	AccessType operator[](IndexType index) const {
		BOLT_ASSERT(device_ptr_ != nullptr);
		return device_ptr_[linearIndex(strides_, index)];
	}

	BOLT_DECL_HOST
	Element getOnHost(IndexType index) const {
		Element tmp;
		BOLT_CHECK(cudaMemcpy(&tmp, &(device_ptr_[linearIndex(strides_, index)]), sizeof(Element), cudaMemcpyDeviceToHost));
		return tmp;
	}

	BOLT_DECL_HOST
	void getOnHost(IndexType index, Element &output_value, cudaStream_t stream) const {
		BOLT_CHECK(cudaMemcpyAsync(&output_value, &(device_ptr_[linearIndex(strides_, index)]), sizeof(Element), cudaMemcpyDeviceToHost, stream));
	}

	// \return Returns true if elements are ordered next to each other in some dimension
	BOLT_DECL_HYBRID
	bool hasContiguousMemory() const {
		return stridesFromSize(this->size()) == strides_;
		// TODO(johny) - support also reordered memory access.
		// return -1 != Find(strides_, 1) || -1 != Find(strides_, -1);
	}

	/// Creates view for part of this view.
	/// \param corner Index of topleft corner
	/// \param size Size of the subimage
	BOLT_DECL_HYBRID
	DeviceImageConstView<TElement, tDimension, Policy>
	subview(const IndexType &corner, const SizeType &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		BOLT_ASSERT(corner >= SizeType());
		BOLT_ASSERT(corner < this->size());
		BOLT_ASSERT((corner + size) <= this->size());
		return DeviceImageConstView<TElement, tDimension, Policy>(
			this->device_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	BOLT_DECL_HYBRID
	DeviceImageConstView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->device_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return DeviceImageConstView<TElement, tDimension - 1, Policy>(
				slice_corner,
				removeDimension(this->size_, tSliceDimension),
				removeDimension(this->strides_, tSliceDimension));
	}

	DeviceImageConstView<TElement, tDimension, Policy> constView() const {
		return *this;
	}
protected:
	// TODO(johny) - view with offsets
	StridesType strides_;
	TElement *device_ptr_;
};


/// View to the part or whole device image, which owns the data.
/// It provides reference access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class DeviceImageView : public DeviceImageConstView<TElement, tDimension, TPolicy> {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = DeviceImageConstView<TElement, tDimension, Policy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element &;

	BOLT_DECL_HYBRID
	DeviceImageView(TElement *device_ptr, SizeType size, SizeType strides) :
		Predecessor(device_ptr, size, strides)
	{}

	BOLT_DECL_HYBRID
	DeviceImageView()
	{}

	/// \return Pointer for raw access to the data buffer.
	BOLT_DECL_HYBRID
	Element *pointer() const {
		return this->device_ptr_;
	}

	BOLT_DECL_DEVICE
	AccessType operator[](IndexType index) const {
		BOLT_ASSERT(this->device_ptr_ != nullptr);
		return this->device_ptr_[linearIndex(this->strides_, index)];
	}

	/// Creates view for part of this view.
	/// \param corner Index of topleft corner
	/// \param size Size of the subimage
	BOLT_DECL_HYBRID
	DeviceImageView<TElement, tDimension, Policy>
	subview(const IndexType &corner, const SizeType &size) const {
		BOLT_ASSERT(corner >= SizeType());
		BOLT_ASSERT(corner < this->size());
		BOLT_ASSERT((corner + size) <= this->size());
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		return DeviceImageView<TElement, tDimension, Policy>(
			this->device_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	BOLT_DECL_HYBRID
	DeviceImageView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("Slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->device_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return DeviceImageView<TElement, tDimension - 1, Policy>(
				slice_corner,
				removeDimension(this->size_, tSliceDimension),
				removeDimension(this->strides_, tSliceDimension));
	}

	// TODO(johny): add DeviceImageConstView(DeviceImageView) constructor and do similar thing for unified image view etc.
	operator DeviceImageConstView<const TElement, tDimension, Policy> () {
		return DeviceImageConstView<const TElement, tDimension, Policy> (this->pointer(), this->size(), this->strides());
	}

	DeviceImageConstView<const TElement, tDimension, Policy> constView() const {
		return DeviceImageConstView<const TElement, tDimension, Policy>(
			this->pointer(), this->size(), this->strides());
	}
};


template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
DeviceImageConstView<const TElement, tDimension, TPolicy>
makeDeviceImageConstView(
	const TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	Vector<typename TPolicy::IndexType, tDimension> strides)
{
	return DeviceImageConstView<const TElement, tDimension, TPolicy>(buffer, size, strides);
}


template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
DeviceImageView<TElement, tDimension, TPolicy>
makeDeviceImageView(
	TElement *buffer,
	Vector<typename TPolicy::IndexType, tDimension> size,
	Vector<typename TPolicy::IndexType, tDimension> strides)
{
	return DeviceImageView<TElement, tDimension, TPolicy>(buffer, size, strides);
}

}  // namespace bolt
