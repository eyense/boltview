// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/math/vector.h>
#include <boltview/mgt/device_code.h>

#ifdef __CUDACC__
#include <boltview/fft/fft_calculator.h>
#include <boltview/device_image.h>
#include <boltview/texture_image.h>
#endif

#include <vector>

namespace bolt {

namespace mgt {

/// \addtogroup mgt Multi GPU Scheduling
/// @{

/// This file contains implementations of memory storage objects for raw memory
/// released from various image types. An image can be converted to its associated
/// memory storage object and vice versa.
/// A memory storage object is able to:
///	- report the size of its owned memory in bytes
///	- check if its owned memory is compatible with an image of requested type and size

/// Helper class for storing image type and size
template<typename TImageType>
class ImageDescriptor {
public:
	using ImageType = TImageType;
	using SizeType = typename TImageType::SizeType;

	explicit ImageDescriptor(SizeType size) : size_(size) {}

	SizeType size() const {
		return size_;
	}

	uint64_t sizeInBytes() const {
		return product(Vector<uint64_t, SizeType::kDimension>(size_)) * sizeof(typename ImageType::Element);
	}

	TImageType allocate() {
		return TImageType(size_);
	}

private:
	SizeType size_;
};

template<typename TDirection>
class FftCalculatorPointer;

#ifdef __CUDACC__
template <int tDim, typename TPolicy>
class ImageDescriptor<FftCalculator<tDim, TPolicy>> {
public:
	using ImageType = FftCalculator<tDim, TPolicy>;
	using SizeType = Vector<int, ImageType::kDataDim>;

	explicit ImageDescriptor(SizeType input_size) :
		input_size(input_size),
		fft_size(getFftImageSize<typename TPolicy::StackPolicy>(input_size)),
		input_strides(stridesFromSize(input_size)),
		fft_strides(stridesFromSize(fft_size))
		{}

	SizeType getInputSize() const {
		return input_size;
	}

	SizeType getFftSize() const {
		return fft_size;
	}

	SizeType getInputStrides() const {
		return input_strides;
	}

	SizeType getFftStrides() const {
		return fft_strides;
	}

	uint64_t sizeInBytes() const {
		return FftCalculator<tDim, TPolicy>().estimateWorkArea(input_size, fft_size);
	}

	ImageType allocate() {
		return ImageType(input_size);
	}

	friend class FftCalculatorPointer<typename TPolicy::Direction>;

private:
	SizeType input_size;
	SizeType input_strides;
	SizeType fft_size;
	SizeType fft_strides;
};

template<typename TElement, int tDim>
using DeviceImageDescriptor = ImageDescriptor<DeviceImage<TElement, tDim>>;

template<typename TElement, int tDim, typename TCudaType = CudaType<TElement>>
using TextureImageDescriptor = ImageDescriptor<TextureImage<TElement, tDim, TCudaType>>;

template<int tDim, typename TPolicy>
using FftCalculatorDescriptor = ImageDescriptor<FftCalculator<tDim, TPolicy>>;

template<typename TElement, int tDimension, typename TCudaType>
cudaChannelFormatDesc getCudaChannelDescriptor(ImageDescriptor<TextureImage<TElement, tDimension, TCudaType>> image_descriptor) {
	return cudaCreateChannelDesc<typename TCudaType::type>();
}
#endif

template<typename TElement, typename TSizeType>
uint64_t getImageSizeInBytesImpl(TSizeType size) {
	return product(Vector<uint64_t, TSizeType::kDimension>(size)) * sizeof(TElement);
}

template<typename TImageType>
uint64_t getImageSizeInBytes(ImageDescriptor<TImageType> image_descriptor) {
	return image_descriptor.sizeInBytes();
}


#ifdef __CUDACC__
template<typename TElement, int tDimension>
uint64_t calculateSizeInBytes(const DeviceImage<TElement, tDimension> &image) {
	return getImageSizeInBytesImpl<TElement>(image.size());
}


template<typename TElement, int tDimension, typename TCudaType>
uint64_t calculateSizeInBytes(const TextureImage<TElement, tDimension, TCudaType> &image) {
	return getImageSizeInBytesImpl<TElement>(image.size());
}


template <int tDim, typename TPolicy>
uint64_t calculateSizeInBytes(const FftCalculator<tDim, TPolicy> &calculator) {
	return calculator.sizeInBytes();
}
#endif


/// Class for storing memory released from DeviceImage. Owns the data.
class DeviceImagePointer {
public:
	DeviceImagePointer() : device_ptr_(nullptr) {}

	#ifdef __CUDACC__
	template<typename TElement, int tDimension>
	explicit DeviceImagePointer(DeviceImage<TElement, tDimension>&& image){
		length_in_bytes_ = calculateSizeInBytes(image);
		device_ptr_.reset(reinterpret_cast<void*>(image.releaseOwnership()));
	}
	#endif

	template<typename TImageType>
	bool isCompatibleWith(ImageDescriptor<TImageType> image_descriptor) const {
		return getImageSizeInBytes(image_descriptor) == length_in_bytes_;
	}

	void* get() const {
		return device_ptr_.get();
	}

	uint64_t size() const {
		return length_in_bytes_;
	}

	template<typename TImageType>
	friend TImageType getImage(DeviceImagePointer && pointer, ImageDescriptor<TImageType> image_descriptor);

private:
	uint64_t length_in_bytes_;
	std::unique_ptr<void, device::VoidPointerDeleter> device_ptr_;
};

template<int tDimension>
bool areEqual(const std::vector<int> vector, Vector<int, tDimension> other) {
	if (tDimension != vector.size()) {
		return false;
	}
	for (int i = 0; i < tDimension; i++) {
		if (vector[i] != other[i]) {
			return false;
		}
	}
	return true;
}


/// Class for storing memory released from TextureImage. Owns the data.
class TextureImagePointer {
public:
	TextureImagePointer() : cuda_array_(nullptr) {}

	#ifdef __CUDACC__
	template<typename TElement, int tDimension, typename TCudaType>
	explicit TextureImagePointer(TextureImage<TElement, tDimension, TCudaType>&& image) : size_(tDimension) {
		auto size = image.size();
		for (int i = 0; i < tDimension; i++) {
			size_[i] = size[i];
		}

		length_in_bytes_ = calculateSizeInBytes(image);
		channel_desc_ = cudaCreateChannelDesc<typename TCudaType::type>();
		cuda_array_.reset(image.releaseOwnership());
	}
	#endif

	template<typename TImageType>
	bool isCompatibleWith(ImageDescriptor<TImageType> image_descriptor) const {
		if (!areEqual(size_, image_descriptor.size())) {
			return false;
		}
		cudaChannelFormatDesc other_desc = getCudaChannelDescriptor(image_descriptor);
		return other_desc.f == channel_desc_.f &&
				other_desc.x == channel_desc_.x &&
				other_desc.y == channel_desc_.y &&
				other_desc.z == channel_desc_.z &&
				other_desc.w == channel_desc_.w;
	}

	cudaArray* get() const {
		return cuda_array_.get();
	}

	uint64_t size() const {
		return length_in_bytes_;
	}

	template<typename TImageType>
	friend TImageType getImage(TextureImagePointer && pointer, ImageDescriptor<TImageType> image_descriptor);

private:
	std::vector<int> size_;
	uint64_t length_in_bytes_;

	cudaChannelFormatDesc channel_desc_;
	std::unique_ptr<cudaArray, device::CudaArrayDeleter> cuda_array_;
};

template<typename TVector>
std::vector<int> toVec(const TVector& vector) {
	return std::vector<int>(vector.pointer(), vector.pointer() + TVector::kDimension);
}

/// Class for storing memory released from FftCalculator. Owns the data.
// NOTE(fidli): TDirection has to be here, because there is difference in inverse/forward plan handle cufftExecC2R(FORWARD_HANDLE) = error,
// and MGT cache does not distinquish between them (FFTCalculatorPointer type would be same for forward/inverse) if not templated here
template<typename TDirection>
class FftCalculatorPointer {
public:
	FftCalculatorPointer() : plan_(nullptr) {}

	template <int tDim, typename TStackPolicy>
	explicit FftCalculatorPointer(FftCalculator<tDim, DeviceFftPolicy<TDirection, TStackPolicy>>&& calculator) {
		length_in_bytes_ = calculateSizeInBytes(calculator);
		plan_.reset(calculator.releaseNativePlan());

		input_size = toVec(calculator.getSpaceDomainSize());
		input_strides = toVec(stridesFromSize(calculator.getSpaceDomainSize()));
		fft_size = toVec(calculator.getFrequencyDomainSize());
		fft_strides = toVec(stridesFromSize(calculator.getFrequencyDomainSize()));
		// NOTE(fidli): tDim is always dimension of fft, regardless of the stack policy
		fftDim = tDim;
		forward = std::is_same<Forward, TDirection>::value;
	}

	template<typename TImageType>
	bool isCompatibleWith(ImageDescriptor<TImageType> image_descriptor) const {
		return fftDim == TImageType::kDimension &&
			areEqual(input_size, image_descriptor.input_size) &&
			areEqual(fft_size, image_descriptor.fft_size) &&
			areEqual(input_strides, image_descriptor.input_strides) &&
			areEqual(fft_strides, image_descriptor.fft_strides);
	}

	DeviceNativePlanType * get() const {
		return plan_.get();
	}

	uint64_t size() const {
		return length_in_bytes_;
	}

	template<typename TDirection2, typename TImageType>
	friend TImageType getImage(FftCalculatorPointer<TDirection2> && pointer, ImageDescriptor<TImageType> image_descriptor);

private:
	uint64_t length_in_bytes_;

	std::vector<int> input_size;
	std::vector<int> input_strides;
	std::vector<int> fft_size;
	std::vector<int> fft_strides;

	int fftDim;

	// NOTE(fidli): type is same for any size, i arbitralily chose Int3
	std::unique_ptr<DeviceNativePlanType, DeviceFftPlan<Int3>::CUFFTDataDestroyer> plan_;
	bool forward;
};

template<typename TImageType>
struct GetSmartPointerHelper;

#ifdef __CUDACC__
template<typename TElement, int tDimension>
struct GetSmartPointerHelper<DeviceImage<TElement, tDimension>> {
	using PointerType = DeviceImagePointer;
};

template<typename TElement, int tDimension, typename TCudaType>
struct GetSmartPointerHelper<TextureImage<TElement, tDimension, TCudaType>> {
	using PointerType = TextureImagePointer;
};

template <int tDim, typename TPolicy>
struct GetSmartPointerHelper<FftCalculator<tDim, TPolicy>> {
	using PointerType = FftCalculatorPointer<typename TPolicy::Direction>;
};
#endif

template<typename TImageType>
typename GetSmartPointerHelper<TImageType>::PointerType getSmartPointer(TImageType && image) {
	return typename GetSmartPointerHelper<TImageType>::PointerType(std::forward<TImageType>(image));
}

template<typename TImageType>
TImageType getImage(DeviceImagePointer && pointer, ImageDescriptor<TImageType> image_descriptor) {
	auto device_ptr = reinterpret_cast<typename TImageType::Element*>(pointer.device_ptr_.release());
	return TImageType(device_ptr, image_descriptor.size(), stridesFromSize(image_descriptor.size()));
}

template<typename TImageType>
TImageType getImage(TextureImagePointer && pointer, ImageDescriptor<TImageType> image_descriptor) {
	return TImageType(pointer.cuda_array_.release(), image_descriptor.size(), stridesFromSize(image_descriptor.size()));
}

template<typename TDirection, typename TImageType>
TImageType getImage(FftCalculatorPointer<TDirection> && pointer, ImageDescriptor<TImageType> image_descriptor) {
	return TImageType(pointer.plan_.release(),
		image_descriptor.getInputSize());
}

/// @}

}  // namespace mgt

}  // namespace bolt
