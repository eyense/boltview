// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

namespace bolt {



#ifdef __CUDACC__
namespace detail {


template<typename TIndex>
BOLT_DECL_DEVICE
TIndex indexFromBlockInfoImplementation(DimensionValue<2> /*tag*/) {
	return TIndex(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
}

template<typename TIndex>
BOLT_DECL_DEVICE
TIndex indexFromBlockInfoImplementation(DimensionValue<3> /*tag*/) {
	return TIndex(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z);
}

}  // namespace detail

/// Return view index from kernel context info.
/// \param view Processed image view
/// \param is_inside_view Output parameter - true if computed index leads inside of the processed view.
template<typename TView>
BOLT_DECL_DEVICE
typename TView::IndexType indexFromBlockInfo(const TView &view, bool *is_inside_view) {
	BOLT_ASSERT(is_inside_view != nullptr);
	auto index = detail::indexFromBlockInfoImplementation<typename TView::IndexType>(DimensionValue<TView::kDimension>());
	*is_inside_view = view.IsIndexInside(index);
	return index;
}

BOLT_DECL_DEVICE
inline int threadIndexFromBlockInfo() {
	return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

#endif  // __CUDACC__

#ifdef __CUDACC__
namespace detail {

template<typename TView>
void distributeBlocksImplementation(const TView &view, dim3 *grid, dim3 *block, DimensionValue<2>) {
	*block = dim3(512, 1, 1);
	*grid = dim3((view.size()[0] + (*block).x - 1) / (*block).x, (view.size()[1] + (*block).y - 1) / (*block).y, 1);
}

template<typename TView>
void distributeBlocksImplementation(const TView &view, dim3 *grid, dim3 *block, DimensionValue<3>) {
	*block = dim3(512, 1, 1);
	*grid = dim3((view.size()[0] + (*block).x - 1) / (*block).x, (view.size()[1] + (*block).y - 1) / (*block).y, (view.size()[2] + (*block).z - 1) / (*block).z);
}

}  // namespace detail

/// Setup grid and block sizes to distribute work on the whole image view
template<typename TView>
void distributeBlocks(const TView &view, dim3 *grid, dim3 *block) {
	detail::distributeBlocksImplementation(view, grid, block, DimensionValue<TView::kDimension>());
}

inline DeviceProperties getDeviceProperties(int device_number) {
	cudaDeviceProp prop;
	BOLT_CHECK(cudaGetDeviceProperties(&prop, device_number));
	DeviceProperties device_properties = {
		prop.totalGlobalMem,
		Int3(prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]),
		Int3(prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]),
		static_cast<int>(prop.texturePitchAlignment)
	};
	return device_properties;
}

inline DeviceMemoryInfo getDeviceMemoryInfo() {
	int device;
	BOLT_CHECK(cudaGetDevice(&device));
	size_t free, total;
	BOLT_CHECK(cudaMemGetInfo(&free, &total));
	DeviceMemoryInfo memory_info {free, total, device};
	return memory_info;
}

inline std::vector<int> getCudaCapableDevices() {
	std::vector<int> result;
	int count = 0;
	cudaDeviceProp prop;
	// get number of devices
	BOLT_CHECK(cudaGetDeviceCount(&count));
	for (int index = 0; index < count; index++) {
		BOLT_CHECK(cudaGetDeviceProperties(&prop, index))
		if (prop.major >= 1) {
			result.push_back(index);
		}
	}
	return result;
}

#endif  // __CUDACC__

// NOTE(fidli): Any view must have defined types: AccessType and IndexType, otherwise the error message is vague (no valid template matches arguments, instead of missing type AccessType etc..)
// TODO(johny) add 'auto' return type when compiling with c++14 or higher
BOLT_HD_WARNING_DISABLE
template<typename TImageView>
BOLT_DECL_HYBRID auto linearAccess(const TImageView &view, int64_t index) -> typename TImageView::AccessType {
	return view[getIndexFromLinearAccessIndex(view, index)];
}


// NOTE(fidli) most of the views begin at 0,0 as top corner - this default case,
// but some views (e.g HalfSpectrumView) does not,
// template-specify this method for desired views to have top corner at different coords
template <typename TView>
BOLT_DECL_HYBRID typename TView::IndexType
topCorner(const TView& /*view*/) {
	using IndexType = typename TView::IndexType;
	return IndexType();
}


BOLT_HD_WARNING_DISABLE
template<typename TImageView>
BOLT_DECL_HYBRID auto getIndexFromLinearAccessIndex(const TImageView &view, int64_t index) -> typename TImageView::IndexType {
	typename TImageView::IndexType coords;
	typename TImageView::IndexType top_corner;
	top_corner = topCorner(view);
	for(int i = 0; i < TImageView::kDimension; ++i) {
		// 1D views have usually int as size, index etc. not Vector<int, 1>
		// Set is used, because coords[i] wont compile
		// meta variable r is used, because it gets implicitly casted to the IndexType element (or int)
		// whereas direct use results to usually cast to long and Set template wont cach int = long...
		auto r = get(top_corner, i);
		r += (index % get(view.size(), i));
		set(coords, i, r);
		index /= get(view.size(), i);
	}
	return coords;
}


template<typename TExtents, typename TCoordinates>
BOLT_DECL_HYBRID int64_t getLinearAccessIndex(
		TExtents extents,
		TCoordinates coordinates)
{
	int dim = TExtents::kDimension;
	int64_t idx = 0;
	int64_t stride = 1;
	for(size_t i = 0; i < dim; ++i) {
		idx += coordinates[i] * stride;
		stride *= extents[i];
	}
	return idx;
}

}  // namespace bolt

