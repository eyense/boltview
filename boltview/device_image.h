// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#if ! defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)


#include <boltview/cuda_utils.h>
#include <boltview/unique_ptr.h>
#include <boltview/device_image_view.h>

namespace bolt {

/// \addtogroup Images
/// @{

/// Device image representation, which owns the data.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class DeviceImage {
public:
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using StridesType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Element = TElement;
	using ElementPointerType = device::unique_ptr<Element>;
	using ViewType = DeviceImageView<TElement, tDimension, Policy>;
	using ConstViewType = DeviceImageConstView<const TElement, tDimension, Policy>;
	static const int kDimension = tDimension;

	DeviceImage() = default;

	explicit DeviceImage(SizeType size) :
		size_(size),
		strides_(stridesFromSize(size)),
		device_ptr_(device::make_unique<Element>(product(Vector<int64_t, SizeType::kDimension>(size))))
	{
		DeviceMemoryInfo mem_info = getDeviceMemoryInfo();
		BOLT_DFORMAT("Device %1% memory %2%/%3% (%4% %% free)",
			mem_info.device,
			mem_info.free_memory,
			mem_info.total_memory,
			100 * mem_info.free_memory / double(mem_info.total_memory));

		BOLT_DFORMAT("Allocated image: %1% bytes per element; size: %2%; strides: %3%",
			sizeof(TElement),
			size_,
			strides_);
	}

	explicit DeviceImage(int size) :
		size_(Int1(size)),
		strides_(stridesFromSize(Int1(size))),
		device_ptr_(device::make_unique<Element>(size))
	{
		static_assert(tDimension == 1, "Only for dimension 1");
		DeviceMemoryInfo mem_info = getDeviceMemoryInfo();
		BOLT_DFORMAT("Device %1% memory %2%/%3% (%4% %% free)",
			mem_info.device,
			mem_info.free_memory,
			mem_info.total_memory,
			100 * mem_info.free_memory / double(mem_info.total_memory));

		BOLT_DFORMAT("Allocated image: %1% bytes per element; size: %2%; strides: %3%",
			sizeof(TElement),
			size_,
			strides_);
	}

	DeviceImage(TIndex width, TIndex height) :
		DeviceImage(SizeType(width, height))
	{
		static_assert(tDimension == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
	}

	DeviceImage(TIndex width, TIndex height, TIndex depth) :
		DeviceImage(SizeType(width, height, depth))
	{
		static_assert(tDimension == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
	}

	DeviceImage(ElementPointerType&& ptr_to_own, SizeType size, SizeType strides) :
		size_(size),
		strides_(strides),
		device_ptr_(ptr_to_own)
	{}

  	DeviceImage(Element* ptr_to_own, SizeType size, SizeType strides) :
  	  	size_(size),
  	  	strides_(strides)
  	{
  	  	device_ptr_.reset(ptr_to_own);
  	}

	DeviceImage(DeviceImage &&other) = default;
	DeviceImage &operator=(DeviceImage &&other) = default;

	~DeviceImage() = default;

	/// Returns the owned data pointer and releases the ownership.
	TElement *releaseOwnership() {
		return device_ptr_.release();
	}

	TElement *pointer() const {
		return device_ptr_.get();
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
		return ViewType(device_ptr_.get(), size_, strides_);
	}

	/// Create view for whole image, which can be used for const access to the image data.
	ConstViewType constView() const {
		return ConstViewType(device_ptr_.get(), size_, strides_);
	}

			#if 0
	/// Create view for whole image, which can be used for const access to the image data.
	::bolt::experimental::DeviceImageConstHalfSpectrumView<const TElement, tDimension> ConstHalfSpectrumView() const {
		return ::bolt::experimental::DeviceImageConstHalfSpectrumView<const TElement, tDimension>(device_ptr_.get(), size_, strides_);
	}

	/// Create view for whole image, which can be used for const access to the image data.
	::bolt::experimental::DeviceImageHalfSpectrumView<TElement, tDimension> HalfSpectrumView(){
		return ::bolt::experimental::DeviceImageHalfSpectrumView<TElement, tDimension>(device_ptr_.get(), size_, strides_);
	}
			#endif

	/// Zeroes the data buffer.
	void clear() {
		BOLT_CHECK(cudaMemset(
				device_ptr_.get(),
				0,
				sizeof(TElement) * static_cast<int64_t>(strides_[tDimension - 1]) * static_cast<int64_t>(size_[tDimension - 1]))
		);
	}

protected:
	SizeType size_;
	StridesType strides_;
	ElementPointerType device_ptr_;
};

/// @}

}  // namespace bolt
