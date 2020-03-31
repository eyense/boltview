// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomaskrupka@eyen.se

#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/texture_image_view.h>
#include <boltview/texture_image_types.h>

namespace bolt {

/// \addtogroup Images
/// @{

/// Texture image representation, it owns data in a cudaArray and provides
/// support for hardware interpolation using CUDA textureObject API
template<typename TElement, int tDimension, typename TCudaType = CudaType<TElement>>
class TextureImage {
public:
	using SizeType = Vector<int, tDimension>;
	using IndexType = Vector<int, tDimension>;
	using CoordinateType = Vector<float, tDimension>;
	using Element = TElement;
	static const int kDimension = tDimension;

	TextureImage() :
		cuda_array_(nullptr)
	{}

	explicit TextureImage(SizeType size)
	{
		allocate(size);
		createTextureObject();
	}

	TextureImage(int width, int height)
	{
		static_assert(tDimension == 2, "Only 2-dimensional images can be specified by 2-dimensional extents!");
		allocate(SizeType(width, height));
		createTextureObject();
	}

	TextureImage(int width, int height, int depth)
	{
		static_assert(tDimension == 3, "Only 3-dimensional images can be specified by 3-dimensional extents!");
		allocate(SizeType(width, height, depth));
		createTextureObject();
	}

	TextureImage(cudaArray *cuda_array, SizeType size, SizeType strides) :
		size_(size),
		strides_(strides),
		cuda_array_(cuda_array)
	{
		createTextureObject();
	}

	~TextureImage() {
		try {
			deallocate();
		} catch (CudaError &e) {
			BOLT_ERROR_FORMAT("Texture image deallocation failure.: %1%", boost::diagnostic_information(e));
		}
	}

	TextureImage(TextureImage &&other) :
		size_(other.size_),
		strides_(other.strides_),
		cuda_array_(other.cuda_array_),
		texture_object_(other.texture_object_)
	{
		other.cuda_array_ = nullptr;
		other.texture_object_ = 0;
	}

	TextureImage &operator=(TextureImage &&other) {
		deallocate();
		size_ = other.size_;
		strides_ = other.strides_;
		cuda_array_ = other.cuda_array_;
		texture_object_ = other.texture_object_;

		other.cuda_array_ = nullptr;
		other.texture_object_ = 0;
		return *this;
	}

	/// Returns the owned cudaArray pointer and releases the ownership.
	cudaArray *releaseOwnership() {
		BOLT_CHECK(cudaDestroyTextureObject(texture_object_));
		cudaArray* cuda_array = cuda_array_;
		cuda_array_ = nullptr;
		return cuda_array;
	}

	/// \return Image size in each dimension.
	SizeType size() const {
		return size_;
	}

	/// \return Offsets needed to increase element coordinate by 1 for each coordinate axis.
	SizeType strides() const {
		return strides_;
	}

	/// Create view for whole image, which can be used for modification of image data.
	TextureImageView<TElement, tDimension, TCudaType> view() {
		return TextureImageView<TElement, tDimension, TCudaType>(cuda_array_, texture_object_, size_, strides_);
	}

	/// Create view for whole image, which can be used for const access to the image data.
	TextureImageConstView<const TElement, tDimension, TCudaType> constView() const {
		return TextureImageConstView<const TElement, tDimension, TCudaType>(cuda_array_, texture_object_, size_, strides_);
	}

protected:
	void allocate(SizeType size) {
		size_ = size;
		strides_ = stridesFromSize(size_);

		Int3 cuda_array_size(
				size[0],
				tDimension > 1 ? size[1] : 0,
				tDimension > 2 ? size[2] : 0);

		BOLT_DFORMAT("arr %1% size %2% strides %3%\n", cuda_array_size, size_, strides_);
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<typename TCudaType::type>();

		try {
			BOLT_CHECK(cudaMalloc3DArray(
				&cuda_array_,
				&channel_desc,
				makeCudaExtent(1, cuda_array_size)));
		} catch (CudaError &e) {
			D_MEM_OUT(sizeof(TElement)*(size[0] * size[1]) * (tDimension == 3 ? size[2] : 1));
			e << getSizeErrorInfo(size);
			throw e;
		}

		BOLT_DFORMAT("Allocated texture image: %1% bytes per element; size: %2%; strides: %3%",
			sizeof(TElement),
			size_,
			strides_);
	}

	void createTextureObject() {
		// create resource descriptor
		cudaResourceDesc resource_descriptor;
		memset(&resource_descriptor, 0, sizeof(cudaResourceDesc));

		resource_descriptor.resType = cudaResourceTypeArray;
		resource_descriptor.res.array.array = cuda_array_;

		// create texture descriptor
		cudaTextureDesc texture_descriptor;
		memset(&texture_descriptor, 0, sizeof(cudaTextureDesc));

		texture_descriptor.normalizedCoords = TCudaType::normalized_coords;
		texture_descriptor.filterMode = TCudaType::filter_mode;
		texture_descriptor.addressMode[0] = TCudaType::address_mode_x;
		texture_descriptor.addressMode[1] = TCudaType::address_mode_y;
		texture_descriptor.addressMode[2] = TCudaType::address_mode_z;
		texture_descriptor.readMode = TCudaType::read_mode;

		BOLT_CHECK(cudaCreateTextureObject(&texture_object_, &resource_descriptor, &texture_descriptor, NULL));
	}

	void deallocate() {
		if (cuda_array_) {
			BOLT_CHECK(cudaFreeArray(cuda_array_));
			BOLT_CHECK(cudaDestroyTextureObject(texture_object_));
		}
	}

	SizeType size_;
	SizeType strides_;

	cudaArray* cuda_array_;
	cudaTextureObject_t texture_object_;
};

/// @}

}  // namespace bolt
