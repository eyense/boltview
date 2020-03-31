// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.eu

#ifndef BOLT_USE_UNIFIED_MEMORY
	#error You have to set BOLT_USE_UNIFIED_MEMORY to ON in CMakeLists.txt to use this
#endif

#pragma once

#if ! defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)


#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>

#include <boltview/cuda_utils.h>
#include <boltview/device_image_view_base.h>
#include <boltview/exceptions.h>
#include <boltview/functors.h>
#include <boltview/view_policy.h>

namespace bolt {

/// \addtogroup Views
/// @{

/// View to the part or whole unified-memory image, which owns the data.
/// It provides only constant access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class UnifiedImageConstView : public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	UnifiedImageConstView(TElement *unified_ptr, SizeType size, SizeType strides) :
		Predecessor(size),
		strides_(strides),
		unified_ptr_(unified_ptr)
	{
		// D_FORMAT("Device view:\n\tsize: %1%\n\tstrides: %2%", size, strides);
	}

	BOLT_DECL_HYBRID
	UnifiedImageConstView() :
		Predecessor(SizeType()),
		unified_ptr_(nullptr)
	{}

	/// \return N-dimensional vector of strides - number of elements,
	/// which must be skipped to increase index by one.
	BOLT_DECL_HYBRID
	StridesType strides()  const {
		return strides_;
	}


	BOLT_DECL_HYBRID
	const Element *pointer() const {
		return unified_ptr_;
	}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		BOLT_ASSERT(unified_ptr_ != nullptr);
		return unified_ptr_[linearIndex(strides_, index)];
	}

	// \return Returns true if elements are ordered next to each other in some dimension
	BOLT_DECL_HYBRID
	bool hasContiguousMemory() const {
		return stridesFromSize(this->size()) == strides_;
	}

	/// Creates view for part of this view.
	/// \param corner Index of topleft corner
	/// \param size Size of the subimage
	UnifiedImageConstView<TElement, tDimension, Policy> subview(const IndexType &corner, const SizeType &size) const {
		// D_FORMAT("Subview:\n\tcorner: %1%\n\tsize: %2%", corner, size);
		BOLT_ASSERT(corner >= SizeType());
		BOLT_ASSERT(corner < this->size());
		BOLT_ASSERT((corner + size) <= this->size());
		return UnifiedImageConstView<TElement, tDimension, Policy>(this->unified_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	UnifiedImageConstView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->unified_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return UnifiedImageConstView<TElement, tDimension - 1, Policy>(
				slice_corner,
				removeDimension(this->size_, tSliceDimension),
				removeDimension(this->strides_, tSliceDimension));
	}

	UnifiedImageConstView<TElement, tDimension, Policy> constView() const {
		return *this;
	}

protected:
	StridesType strides_;
	TElement *unified_ptr_;
};


/// View to the part or whole unified-memory image, which owns the data.
/// It provides reference access to the data. It is usable on both host/device sides.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class UnifiedImageView : public UnifiedImageConstView<TElement, tDimension, TPolicy> {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = UnifiedImageConstView<TElement, tDimension, Policy>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element &;

	BOLT_DECL_HYBRID
	UnifiedImageView(TElement *unified_ptr, SizeType size, SizeType strides) :
		Predecessor(unified_ptr, size, strides)
	{}

	/// \return Pointer for raw access to the data buffer.
	BOLT_DECL_HYBRID
	Element *pointer() const {
		return this->unified_ptr_;
	}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		BOLT_ASSERT(this->unified_ptr_ != nullptr);
		return this->unified_ptr_[linearIndex(this->strides_, index)];
	}

	/// Creates view for part of this view.
	/// \param corner Index of topleft corner
	/// \param size Size of the subimage
	UnifiedImageView<TElement, tDimension, Policy> subview(const IndexType &corner, const SizeType &size) const {
		BOLT_ASSERT(corner >= SizeType());
		BOLT_ASSERT(corner < this->size());
		BOLT_ASSERT((corner + size) <= this->size());
		return UnifiedImageView<TElement, tDimension, Policy>(this->unified_ptr_ + linearIndex(this->strides_, corner), size, this->strides_);
	}

	/// Creates view for cut through the image
	/// \tparam tSliceDimension Dimension axis perpendicular to the cut
	/// \param slice Coordinate of the slice - index in tSliceDimension
	template<int tSliceDimension>
	UnifiedImageView<TElement, tDimension - 1, Policy> slice(int slice) const {
		// D_FORMAT("slice:\n\tdimension: %1%\n\tslice: %2%", tSliceDimension, slice);
		static_assert(tSliceDimension < tDimension, "Wrong slicing dimension");
		static_assert(tSliceDimension >= 0, "Wrong slicing dimension");
		BOLT_ASSERT(slice >= 0);
		BOLT_ASSERT(slice < this->size()[tSliceDimension]);
		TElement *slice_corner = this->unified_ptr_ + int64_t(this->strides_[tSliceDimension]) * slice;
		return UnifiedImageView<TElement, tDimension - 1, Policy>(
				slice_corner,
				removeDimension(this->size_, tSliceDimension),
				removeDimension(this->strides_, tSliceDimension));
	}

	operator UnifiedImageConstView<const TElement, tDimension, Policy> () {
		return UnifiedImageConstView<const TElement, tDimension, Policy> (this->pointer(), this->size(), this->strides());
	}

	UnifiedImageConstView<const TElement, tDimension, Policy> constView() const {
		return UnifiedImageConstView<const TElement, tDimension, Policy>(
			this->pointer(), this->size(), this->strides());
	}
};

template<typename View>
void prefetchView(View view, int dstDevice, cudaStream_t stream = 0){
	int elementSize = sizeof(*(view.pointer()));

	if(view.hasContiguousMemory()){
		cudaMemPrefetchAsync(
				view.pointer(),
				view.elementCount()*elementSize,
				dstDevice,
				stream);
	}

	// Non contiguous memory prefetch too slow

	// else{
	// 	const int columns = view.size()[0];
	// 	const int lines = view.size()[1];
	// 	const int stride = view.strides()[1];
	//
	// 	for(int i = 0; i < lines; ++i){
	// 		cudaMemPrefetchAsync(
	// 			view.pointer() + i*stride,
	// 			columns * elementSize,
	// 			dstDevice,
	// 			stream);
	// 	}
	// }
}

/// @}

}  // namespace bolt
