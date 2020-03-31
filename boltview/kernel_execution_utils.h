#pragma once

 #include <boost/type_traits.hpp>

//TODO(johny) - not used - remove
namespace bolt {

/** \addtogroup Utilities
 * @{
 **/

template<typename TImageSize>
dim3
computeGridSize(const dim3 &aBlockSize, const TImageSize &aImageSize)
{
	Vector<int, 3> grid(1, 1, 1);
	Vector<int, 3> block(aBlockSize.x, aBlockSize.y, aBlockSize.z);

	for (int i = 0; i < dimension<TImageSize>::value; ++i) {
		grid[i] = (aImageSize[i] + block[i] - 1) / block[i];
	}
	return dim3(grid[0], grid[1], grid[2]);
}

template<typename TCoords>
BOLT_DECL_DEVICE TCoords
cornerCoordsFromBlockDim()
{
	Vector<int, 3> index(blockIdx.x, blockIdx.y, blockIdx.z);
	Vector<int, 3> size(blockDim.x, blockDim.y, blockDim.z);
	TCoords result;
	for (int i = 0; i < dimension<TCoords>::value; ++i) {
		result[i] = index[i] * size[i];
	}
	return result;
}

template<typename TCoords>
BOLT_DECL_DEVICE TCoords
coordsFromBlockDim()
{
	Vector<int, 3> thread_index(threadIdx.x, threadIdx.y, threadIdx.z);
	Vector<int, 3> index(blockIdx.x, blockIdx.y, blockIdx.z);
	Vector<int, 3> size(blockDim.x, blockDim.y, blockDim.z);
	TCoords result;
	for (int i = 0; i < TCoords::kDimension; ++i) {
		result[i] = index[i] * size[i] + thread_index[i];
	}
	return result;
}


/**
 * @}
 **/


}//namespace bolt

