// Copyright 2016 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se


#pragma once

#include <boltview/image_locator.h>

namespace bolt {

namespace detail {

#ifdef __CUDACC__

/// use this struct as a __shared__ variable in CUDA kernels for effective loading of data,
/// size of TElement has to be a multiple of 4 bytes
template<typename TElement, int tThreadCount, int tChunkLoadsPerThread = 4>
struct SharedMemoryBuffer {
	static constexpr int kNumberOfChunksPerElement = sizeof(TElement) / sizeof(int32_t);
	static constexpr int kNumberOfElements = tChunkLoadsPerThread * tThreadCount / kNumberOfChunksPerElement;
	static constexpr int kNumberOfChunks = kNumberOfElements * kNumberOfChunksPerElement;

	static_assert(sizeof(TElement) % 4 == 0, "TElement is not loadable in 4byte chunks.");

	template<typename TView>
	BOLT_DECL_DEVICE
	void preloadFull(const TView &view, int start) {
		PreloadPartial(view, start, kNumberOfElements);
	}

	template<typename TView>
	BOLT_DECL_DEVICE
	void preloadPartial(TView view, int start, const int count) {
		const TElement *element = view.pointer() + start;
		const int32_t *source_pointer = reinterpret_cast<const int32_t *>(element);
		int thread_index = threadIndexFromBlockInfo();
		while (thread_index < count * kNumberOfChunksPerElement) {
			elements[thread_index] = source_pointer[thread_index];
			thread_index += tThreadCount;
		}
	}

	BOLT_DECL_DEVICE
	TElement &get(int index) {
		return *reinterpret_cast<TElement *>(elements + index * kNumberOfChunksPerElement);
	}

private:
	int32_t elements[kNumberOfChunks];
};



/// \return View coordinates for thread
template<int tDimension>
BOLT_DECL_DEVICE
inline Vector<int, tDimension> getViewCoordsInBlock();

template<>
BOLT_DECL_DEVICE
inline Vector<int, 2> getViewCoordsInBlock(){ return Vector<int, 2>(threadIdx.x, threadIdx.y); }

template<>
BOLT_DECL_DEVICE
inline Vector<int, 3> getViewCoordsInBlock(){ return Vector<int, 3>(threadIdx.x, threadIdx.y, threadIdx.z); }


/// \return Index into 1-dimensional array
BOLT_DECL_DEVICE
inline
int getIndexInBlock() { return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; }


/// \return Coordinates of beginning of the block
template<int tDimension>
BOLT_DECL_DEVICE
inline Vector<int, tDimension> getBlockPosition();

template<>
BOLT_DECL_DEVICE
inline Vector<int, 2> getBlockPosition(){
	return Vector<int, 2>(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y);
}

template<>
BOLT_DECL_DEVICE
inline Vector<int, 3> getBlockPosition(){
	return Vector<int, 3>(blockDim.x * blockIdx.x, blockDim.y * blockIdx.y, blockDim.z * blockIdx.z);
}


/// \return Coordinates in view for given 1-dimensional index
template<int tDimension>
BOLT_DECL_DEVICE
inline Vector<int, tDimension> getCoordsInView(int index, Vector<int, tDimension> overlap);


/// \return Coordinates from given 1-dimensional index
template<>
BOLT_DECL_DEVICE
inline Vector<int, 2> getCoordsInView(int index, Vector<int, 2> overlap){
	int sizeX = blockDim.x + overlap[0];
	return Vector<int, 2>(index % sizeX, index / sizeX);
}

template<>
BOLT_DECL_DEVICE
inline Vector<int, 3> getCoordsInView(int index, Vector<int, 3> overlap){
	int sizeX = blockDim.x + overlap[0];
	int sizeY = blockDim.y + overlap[1];

	int indexZ = index / (sizeX * sizeY);
	index -= indexZ * sizeX * sizeY;
	return Vector<int, 3>(index % sizeX, index / sizeX, indexZ);
}

template<int tDimension>
BOLT_DECL_DEVICE
inline Vector<int, tDimension> getSharedMemoryViewSize(Vector<int, tDimension> overlap);

template<>
BOLT_DECL_DEVICE
inline Vector<int, 2> getSharedMemoryViewSize(Vector<int, 2> overlap){
	return Vector<int, 2>(blockDim.x + overlap[0], blockDim.y + overlap[1]);
}

template<>
BOLT_DECL_DEVICE
inline Vector<int, 3> getSharedMemoryViewSize(Vector<int, 3> overlap){
	return Vector<int, 3>(blockDim.x + overlap[0], blockDim.y + overlap[1], blockDim.z + overlap[2]);
}

/// Load view and overlaping parts into \param data
template<typename TView, typename TPolicy>
BOLT_DECL_DEVICE
void loadToSharedMemory(TView view, TPolicy policy,	typename TView::Element data[]){
	auto overlap = policy.overlapStart + policy.overlapEnd;
	int index = getIndexInBlock();
	int toPreload = product(getSharedMemoryViewSize(overlap));

	auto block_position = getBlockPosition<TView::kDimension>();
	auto locator = LocatorConstruction<TPolicy::kBorderHandling>::create(view, block_position - policy.overlapStart);

	int loopStride = product(dim3ToInt<3>(blockDim));

	//TODO(johny) - check
	//Jarda: I have add the overlap here.
	// I am confident that it should be this way,
	// but the code was somehow working without overlap if all dimensions were grearer then block size. I do not know why...
	auto extent = view.size() + overlap;

	for(int i = index; i < toPreload; i += loopStride){
		auto pos = getCoordsInView(i, overlap);
		if (pos < extent){
			data[i] =locator[pos];
		}
	}
}

template<typename TElement, int tDimension>
BOLT_DECL_DEVICE
DeviceImageConstView<const TElement, tDimension>
makeViewForSharedMemoryBuffer(const TElement *buffer, Vector<int, tDimension> size, Vector<int, tDimension> strides) {
	return DeviceImageConstView<const TElement, tDimension>(buffer, size, strides);
}


#endif  // __CUDACC__
}  // namespace detail

}  // namespace bolt
