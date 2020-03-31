// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.eu

#ifndef BOLT_USE_UNIFIED_MEMORY
	#error You have to set BOLT_USE_UNIFIED_MEMORY to ON in CMakeLists.txt to use this
#endif

#if ! defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)


#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/unified_image_view.h>

namespace bolt {

/// \addtogroup Images
/// @{

/// Unified-memory image representation, which owns the data.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class UnifiedImage {
public:
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Element = TElement;
	using ViewType = UnifiedImageView<TElement, tDimension, Policy>;
	using ConstViewType = UnifiedImageConstView<const TElement, tDimension, Policy>;
	static const int kDimension = tDimension;

	UnifiedImage() :
		unified_ptr_(nullptr)
	{}

	explicit UnifiedImage(SizeType size)
	{
		reallocate(size);
	}

	UnifiedImage(int width, int height)
	{
		static_assert(tDimension == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
		reallocate(SizeType(width, height));
	}

	UnifiedImage(int width, int height, int depth)
	{
		static_assert(tDimension == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
		reallocate(SizeType(width, height, depth));
	}

	~UnifiedImage() {
		try {
			deallocate();
		} catch (CudaError &e) {
			BOLT_ERROR_FORMAT("Unified image deallocation failure.: %1%", boost::diagnostic_information(e));
		}
	}

	UnifiedImage(UnifiedImage &&other) :
		size_(other.size_),
		strides_(other.strides_),
		unified_ptr_(other.unified_ptr_)
	{
		other.unified_ptr_ = nullptr;
	}

	UnifiedImage &operator=(UnifiedImage &&other) {
		deallocate();
		size_ = other.size_;
		strides_ = other.strides_;
		unified_ptr_ = other.unified_ptr_;

		other.unified_ptr_ = nullptr;
		return *this;
	}

	TElement *pointer() const {
		return unified_ptr_;
	}

	/// \return Image size in each dimension.
	SizeType size() const {
		return size_;
	}

	/// \return Offsets needed to increase element coordinate by 1 for each coordinate axis.
	StridesType strides() const {
		return strides_;
	}

	/// Create view for whole image, which can be used for modification of image data.
	ViewType view() {
		return ViewType(unified_ptr_, size_, strides_);
	}

	/// Create view for whole image, which can be used for const access to the image data.
	ConstViewType constView() const {
		return ConstViewType(unified_ptr_, size_, strides_);
	}

	/// Zeroes the data buffer.
	void clear() {
		BOLT_CHECK(cudaMemset(
				unified_ptr_,
				0,
				sizeof(TElement) * static_cast<int64_t>(strides_[tDimension - 1]) * static_cast<int64_t>(size_[tDimension - 1])));
	}

protected:
	void reallocate(SizeType size) {
		uint64_t num_elements = product(Vector<uint64_t, SizeType::kDimension>(size));
		BOLT_CHECK(cudaMallocManaged(&unified_ptr_, sizeof(TElement) * num_elements));
		cudaDeviceSynchronize();
		size_ = size;
		strides_ = stridesFromSize(size_);

		DeviceMemoryInfo mem_info = getDeviceMemoryInfo();
		BOLT_DFORMAT("Unified %1% memory %2%/%3% (%4% %% free)",
			mem_info.device,
			mem_info.free_memory,
			mem_info.total_memory,
			100 * mem_info.free_memory / double(mem_info.total_memory));

		BOLT_DFORMAT("Allocated image: %1% bytes per element; size: %2%; strides: %3%",
			sizeof(TElement),
			size_,
			strides_);
	}

	void deallocate() {
		if (unified_ptr_) {
			BOLT_CHECK(cudaDeviceSynchronize()); // Unified memory could be accessed by running kernel
			BOLT_CHECK(cudaFree(unified_ptr_));
		}
	}

	SizeType size_;
	StridesType strides_;
	TElement *unified_ptr_;
};

/// @}

}  // namespace bolt
