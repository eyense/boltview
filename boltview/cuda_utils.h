// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#endif  // __CUDACC__

#include <iostream>
#include <type_traits>
#include <vector>

#include <boost/format.hpp>

#include <boltview/cuda_defines.h>
#include <boltview/exceptions.h>
#include <boltview/math/vector.h>
#include <boltview/device_properties.h>

#include <boltview/view_traits.h>

#define D_MEM_OUT(sizeInBytes) /* NOLINT */ \
	{ \
		size_t freeSpace; \
		size_t totalSpace; \
		::std::string msg = ::std::string("Tried to allocate: ") + ::std::to_string((sizeInBytes)/(1024*1024)) + ::std::string(" MB."); \
		if(cudaMemGetInfo(&freeSpace, &totalSpace) == cudaSuccess){ \
				msg += ::std::string(" When only ") + ::std::to_string(freeSpace/(1024*1024)) + ::std::string(" MB out of ") + ::std::to_string(totalSpace/(1024*1024)) + ::std::string(" MB was available."); \
		} \
		BOLT_DFORMAT(msg); \
	} \

/// Utilities for conversion between cuda defined data structures and more convenient representations.
/// Utilities for strided memory access.

namespace bolt {

struct Byte {
	static constexpr int kBytes = 1;
};

template<typename TType>
struct MemoryUnit {
	static constexpr int kBytes = sizeof(TType);
};

template<typename TUnit>
struct Stride {
	using Unit = TUnit;

	int64_t value;
};

using ByteStride = Stride<Byte>;

/// \addtogroup Utilities
/// @{

/// \addtogroup CudaConversions
/// @{
#ifdef __CUDACC__
/// \param bytes Size of array element in bytes.
/// \param size Size of 1D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, int64_t size) {
	return make_cudaExtent(bytes * size, 1, 1);
}

/// \param bytes Size of array element in bytes.
/// \param size Size of 1D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, Vector<int, 1> size) {
	return make_cudaExtent(bytes * size[0], 1, 1);
}

/// \param bytes Size of array element in bytes.
/// \param size Size of 2D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, Int2 size) {
	return make_cudaExtent(bytes * size[0], size[1], 1);
}

/// \param bytes Size of array element in bytes.
/// \param size Size of 3D array in number of elements.
inline cudaExtent makeCudaExtent(int bytes, Int3 size) {
	return make_cudaExtent(bytes * size[0], size[1], size[2]);
}

// /// \param bytes Size of array element in bytes.
// /// \param size Size of 1D array in number of elements.
// inline cudaExtent makeCudaExtent(int64_t bytes, int64_t size) {
// 	return make_cudaExtent(bytes * size, 1, 1);
// }
//
// /// \param bytes Size of array element in bytes.
// /// \param size Size of 1D array in number of elements.
// inline cudaExtent makeCudaExtent(int64_t bytes, Vector<int64_t, 1> size) {
// 	return make_cudaExtent(bytes * size[0], 1, 1);
// }

/// \param bytes Size of array element in bytes.
/// \param size Size of 2D array in number of elements.
inline cudaExtent makeCudaExtent(int64_t bytes, LongInt2 size) {
	return make_cudaExtent(bytes * size[0], size[1], 1);
}

/// \param bytes Size of array element in bytes.
/// \param size Size of 3D array in number of elements.
inline cudaExtent makeCudaExtent(int64_t bytes, LongInt3 size) {
	return make_cudaExtent(bytes * size[0], size[1], size[2]);
}

inline cudaPos makeCudaPos(Int3 pos) {
	return make_cudaPos(pos[0], pos[1], pos[2]);
}

inline cudaPos makeCudaPos(Int2 pos) {
	return make_cudaPos(pos[0], pos[1], 0);
}

inline cudaPos makeCudaPos(int pos) {
	return make_cudaPos(pos, 0, 0);
}

/// Convert cuda pitched pointer to element based strides.
/// \param bytes Size of element in bytes.
/// \param pitched_ptr Actual pitched pointer.
template<int tDimension>
Vector<int, tDimension> pitchedPtrToStrides(int bytes, cudaPitchedPtr pitched_ptr);

template<>
inline Vector<int, 2> pitchedPtrToStrides<2>(int bytes, cudaPitchedPtr pitched_ptr) {
	BOLT_ASSERT(pitched_ptr.pitch % bytes == 0);
	return Vector<int, 2>(1, pitched_ptr.pitch / bytes);
}

template<>
inline Vector<int, 3> pitchedPtrToStrides<3>(int bytes, cudaPitchedPtr pitched_ptr) {
	BOLT_ASSERT(pitched_ptr.pitch % bytes == 0);
	return Vector<int, 3>(1, pitched_ptr.pitch / bytes, (pitched_ptr.pitch / bytes) * pitched_ptr.ysize);
}

/// Convert element based strides to cuda pitched pointer.
/// \param ptr Data pointer
/// \param size Size of image buffer
/// \param strides Strides are unused - here only for the compatible overload and additional check that strides are set to 1.
template<typename TElement>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, int64_t size, int64_t strides) {
	BOLT_ASSERT(strides == 1 && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), sizeof(TElement) * size, size, 1);
}

template<typename TElement, typename TSizePrecision, typename TStridesPrecision>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, Vector<TSizePrecision, 1> size, Vector<TStridesPrecision, 1> strides) {
	BOLT_ASSERT(strides[0] == 1 && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), sizeof(TElement) * size[0], size[0], 1);
}

template<typename TElement, typename TSizePrecision, typename TStridesPrecision>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, Vector<TSizePrecision, 2> size, Vector<TStridesPrecision, 2> strides) {
	BOLT_ASSERT(strides[0] == 1 && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), sizeof(TElement) * strides[1], size[0], size[1]);
}

template<typename TElement, typename TSizePrecision, typename TStridesPrecision>
inline cudaPitchedPtr stridesToPitchedPtr(TElement *ptr, Vector<TSizePrecision, 3> size, Vector<TStridesPrecision, 3> strides) {
	BOLT_ASSERT(strides[0] == 1 && "Pitched cuda pointer is usable only for continuous mmemory blocks");
	// pitched pointer wraps only void * -> goodbye const correctness here
	return make_cudaPitchedPtr(const_cast<void *>(reinterpret_cast<const void *>(ptr)), sizeof(TElement) * strides[1], size[0], size[1]);
}

// Convert dim3 to Int2 or Int3
template<int tDimension>
BOLT_DECL_HYBRID
Vector<int, tDimension> dim3ToInt(dim3 dimension);

template<>
BOLT_DECL_HYBRID
inline Vector<int, 2> dim3ToInt(dim3 dim){
	BOLT_ASSERT(dim.z == 1);
	return Vector<int, 2>(dim.x, dim.y);
}

template<>
BOLT_DECL_HYBRID
inline Vector<int, 3> dim3ToInt(dim3 dim){
	return Vector<int, 3>(dim.x, dim.y, dim.z);
}

/// @}

/// \addtogroup IndexMapping
/// @{

/// Return view index from kernel context info.
/// \param view Processed image view
/// \param is_inside_view Output parameter - true if computed index leads inside of the processed view.
template<typename TView>
BOLT_DECL_DEVICE
typename TView::IndexType indexFromBlockInfo(const TView &view, bool *is_inside_view);

/// Setup grid and block sizes to distribute work on the whole image view
template<typename TView>
void distributeBlocks(const TView &view, dim3 *grid, dim3 *block);

/// get 1 to 1 mapping between cuda threads in grid and view coordinates
template<int tDimension>
BOLT_DECL_DEVICE Vector<int, tDimension>
mapBlockIdxAndThreadIdxToViewCoordinates();


template<>
BOLT_DECL_DEVICE Vector<int, 1>
inline mapBlockIdxAndThreadIdxToViewCoordinates<1>()
{
	BOLT_ASSERT(gridDim.y == 1);
	BOLT_ASSERT(gridDim.z == 1);
	BOLT_ASSERT(blockDim.y == 1);
	BOLT_ASSERT(blockDim.z == 1);
	BOLT_ASSERT(threadIdx.y == 0);
	BOLT_ASSERT(threadIdx.z == 0);
	return Vector<int, 1>(blockIdx.x * blockDim.x + threadIdx.x);
}

template<>
BOLT_DECL_DEVICE Vector<int, 2>
inline mapBlockIdxAndThreadIdxToViewCoordinates<2>()
{
	BOLT_ASSERT(gridDim.z == 1);
	BOLT_ASSERT(blockDim.z == 1);
	BOLT_ASSERT(threadIdx.z == 0);
	return Vector<int, 2>(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

template<>
BOLT_DECL_DEVICE Vector<int, 3>
inline mapBlockIdxAndThreadIdxToViewCoordinates<3>()
{
	return Vector<int, 3>(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
}


BOLT_DECL_DEVICE
inline int threadOrderFromIndex() {
	return threadIdx.x
		+ threadIdx.y * blockDim.x
		+ threadIdx.z * blockDim.x * blockDim.y;
}

BOLT_DECL_DEVICE
inline int currentBlockSize() {
	return blockDim.x * blockDim.y * blockDim.z;
}

/// @}

#endif  // __CUDACC__

/// \return Strides for memory without padding.
template<typename TVector>
BOLT_DECL_HYBRID
TVector stridesFromSize(TVector size) {
	TVector stride;
	stride[0] = 1;
	for (int i = 1; i < TVector::kDimension; ++i) {
		stride[i] = stride[i-1] * size[i-1];
	}
	return stride;
}


/// \return Strides for memory without padding.
template<>
BOLT_DECL_HYBRID
inline Int2 stridesFromSize<Int2>(Int2 size) {
	return { 1, size[0] };
}

/// \return Strides for memory without padding.
BOLT_DECL_HYBRID
inline int stridesFromSize(int /*size*/) {
	return 1;
}

/// \return Strides for memory without padding.
BOLT_DECL_HYBRID
inline Int1 stridesFromSize(Int1 /*size*/) {
	return { 1 };
}

/// \return Strides for memory without padding.
template<>
BOLT_DECL_HYBRID
inline Int3 stridesFromSize<Int3>(Int3 size) {
	return { 1, size[0], size[0] * size[1] };
}


namespace detail {
inline bool checkStrides(int stride, int64_t size) {
	return stride == 1;
}

template<typename TVector, typename TSize>
inline bool checkStrides(TVector strides, TSize size) {
	return strides[2] == (strides[1] * size[1]);
}

}

/// \return false if there is memory gap between slices
BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
inline bool isContinuousInZ(TView view) {
	static_assert(IsImageView<TView>::value || IsArrayView<TView>::value, "Works only for image and array views.");
	return TView::kDimension < 3 || detail::checkStrides(view.strides(), view.size());
}


/// Dimension type wrapper for tag dispatch
template<int tDimension>
struct DimensionValue {
	static const int kDimension = tDimension;
};


/// Access image view element by 1D index - order of the element.
template<typename TImageView>
BOLT_DECL_HYBRID
auto linearAccess(const TImageView &view, int64_t index) -> typename TImageView::AccessType;

template<typename TImageView>
BOLT_DECL_HYBRID
auto getIndexFromLinearAccessIndex(const TImageView &view, int64_t index) -> typename TImageView::IndexType;


/// get order of the element from size and coordinates.
template<typename TExtents, typename TCoordinates>
BOLT_DECL_HYBRID
int64_t getLinearAccessIndex(TExtents extents, TCoordinates coordinates);


/// Declval for both device and host (necessary for clang)
/// \return Cannot be called.
template<class TT>
BOLT_DECL_HYBRID
typename std::add_rvalue_reference<TT>::type declval() noexcept;

DeviceProperties getDeviceProperties(int device_number);

DeviceMemoryInfo getDeviceMemoryInfo();

std::vector<int> getCudaCapableDevices();

/// @}

// TODO(johny) - handle cuda streams in non-cuda code differently
#if !defined(__CUDACC__)
using cudaStream_t = void*;
#endif  // __CUDACC__

}  // namespace bolt

#include "cuda_utils.tcc"
