// Copyright 2017 Eyen SE
// Author: Tomas Krupka tomaskrupka@eyen.se

#pragma once

#if !defined(__CUDACC__)
#error "This header can be included only into sources compiled by nvcc."
#endif  // !defined(__CUDACC__)

#include <boltview/cuda_defines.h>

namespace bolt {

// The following struct is used to set filtering parameters for a cudaTextureObject_t wrapped in our TextureImage implementation,
// possible values and exact effects of these parameters are described in the Cuda Programming Guide (Texture Object API, Texture Fetching)
// see CustomFilteringTypeTest in unit tests to see how tu use custom filtering parameters
struct CudaTypeDefaults {
	static const bool normalized_coords = false; // address image via coordinates in range (-0.5,0.5)
	static const cudaTextureFilterMode filter_mode = cudaFilterModeLinear;
	static const cudaTextureReadMode read_mode = cudaReadModeElementType;
	static const cudaTextureAddressMode address_mode_x = cudaAddressModeBorder;
	static const cudaTextureAddressMode address_mode_y = cudaAddressModeBorder;
	static const cudaTextureAddressMode address_mode_z = cudaAddressModeBorder;
	// static const cudaTextureAddressMode address_mode_x = cudaAddressModeClamp;
	// static const cudaTextureAddressMode address_mode_y = cudaAddressModeClamp;
	// static const cudaTextureAddressMode address_mode_z = cudaAddressModeClamp;

	// if using unnormalized coordinates, we have to add 0.5, to get the actual values,
	// because cuda treats data samples as being in center of pixels/voxels
	BOLT_DECL_DEVICE
	static constexpr float offset() {
		return normalized_coords ? 0.0f : 0.5f;
	}
};

template<typename TType>
struct CudaType : public CudaTypeDefaults {};

template<>
struct CudaType<int> : public CudaTypeDefaults {
	using type = int1;
	static const cudaTextureFilterMode filter_mode = cudaFilterModePoint;
	BOLT_DECL_DEVICE
	static int toVec(const int1& vec) {
		return vec.x;
	}
};

template<>
struct CudaType<Int2> : public CudaTypeDefaults {
	using type = int2;
	static const cudaTextureFilterMode filter_mode = cudaFilterModePoint;
	BOLT_DECL_DEVICE
	static Int2 toVec(const int2& vec) {
		return Int2(vec.x, vec.y);
	}
};

template<>
struct CudaType<Int4> : public CudaTypeDefaults {
	using type = int4;
	static const cudaTextureFilterMode filter_mode = cudaFilterModePoint;
	BOLT_DECL_DEVICE
	static Int4 toVec(const int4& vec) {
		return Int4(vec.x, vec.y, vec.z, vec.w);
	}
};

template<>
struct CudaType<float> : public CudaTypeDefaults {
	using type = float1;
	BOLT_DECL_DEVICE
	static float toVec(const float1& vec) {
		return vec.x;
	}
};

template<>
struct CudaType<Float2> : public CudaTypeDefaults {
	using type = float2;
	BOLT_DECL_DEVICE
	static Float2 toVec(const float2& vec) {
		return Float2(vec.x, vec.y);
	}
};

template<>
struct CudaType<Float4> : public CudaTypeDefaults {
	using type = float4;
	BOLT_DECL_DEVICE
	static Float4 toVec(const float4& vec) {
		return Float4(vec.x, vec.y, vec.z, vec.w);
	}
};

template<class TElement>
using DefaultCudaType = CudaType<typename std::remove_const<TElement>::type>;

}  // namespace bolt
