// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <boltview/image_view_utils.h>
#include <boltview/view_traits.h>
#include <boltview/exception_error_info.h>

namespace bolt {


#ifdef __CUDACC__
template <typename TFromView, typename TToView>
BOLT_GLOBAL void copyKernel(
	TFromView from_view,
	TToView to_view)
{
	int element_count = from_view.elementCount();
	int tid = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tid;
	int grid_size = blockDim.x * gridDim.x;

	while (index < element_count) {
		linearAccess(to_view, index) = linearAccess(from_view, index);
		index += grid_size;
	}
	__syncthreads();
}


template <typename TFromView, typename TToView>
void copyDeviceToDeviceAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	// TODO(johny) - use memcpy for memory based views
	constexpr int kBucketSize = 4;  // Bundle more computation in one block

	dim3 block(512, 1, 1);
	dim3 grid(1 + (from_view.elementCount() - 1) / (block.x * kBucketSize), 1, 1);

	copyKernel<TFromView, TToView><<<grid, block, 0, cuda_stream>>>(from_view, to_view);
	BOLT_CHECK_ERROR_STATE("After copyKernel");
}
#endif  // __CUDACC__

template<int tDimension, bool tMemcpyAble>
struct CopyHostToHostHelper;


template <>
struct CopyHostToHostHelper<1, true> {
	template <typename TFromView, typename TToView>
	static inline void copyHostToHost(TFromView from_view, TToView to_view) {
		for (int64_t i = 0; i < to_view.size(); ++i) {
			to_view[Int1(i)] = from_view[Int1(i)];
		}
	}
};

template<>
struct CopyHostToHostHelper<2, true> {
	template <typename TFromView, typename TToView>
	static inline void copyHostToHost(TFromView from_view, TToView to_view) {
		auto from_ptr = from_view.pointer();
		auto from_strides = from_view.strides();
		auto to_ptr = to_view.pointer();
		auto to_strides = to_view.strides();
		auto size = from_view.size();
		for (int64_t j = 0; j < size[1]; ++j) {
			for (int64_t i = 0; i < size[0]; ++i) {
				to_ptr[i * to_strides[0] + j * to_strides[1]] =
					from_ptr[i * from_strides[0] + j * from_strides[1]];
			}
		}
	}
};

template<>
struct CopyHostToHostHelper<3, true> {
	template <typename TFromView, typename TToView>
	static inline void copyHostToHost(TFromView from_view, TToView to_view) {
		auto from_ptr = from_view.pointer();
		auto from_strides = from_view.strides();
		auto to_ptr = to_view.pointer();
		auto to_strides = to_view.strides();
		auto size = from_view.size();
		for (int64_t k = 0; k < size[2]; ++k) {
			for (int64_t j = 0; j < size[1]; ++j) {
				for (int64_t i = 0; i < size[0]; ++i) {
					to_ptr[i * to_strides[0] + j * to_strides[1] + k * to_strides[2]] =
						from_ptr[i * from_strides[0] + j * from_strides[1] + k * from_strides[2]];
				}
			}
		}
	}
};

template<int tDimension>
struct CopyHostToHostHelper<tDimension, false> {
	template <typename TFromView, typename TToView>
	static inline void copyHostToHost(TFromView from_view, TToView to_view) {
		for (int64_t i = 0; i < from_view.elementCount(); ++i) {
			linearAccess(to_view, i) = linearAccess(from_view, i);
		}
	}
};

template <typename TFromView, typename TToView>
inline void copyHostToHost(
	TFromView from_view,
	TToView to_view)
{
	CopyHostToHostHelper<TFromView::kDimension, IsMemcpyAble<TFromView>::value && IsMemcpyAble<TToView>::value>::copyHostToHost(from_view, to_view);
}

#ifdef __CUDACC__
template <typename TFromView, typename TToView>
void copyBySlices(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<2>)
{
	BOLT_ASSERT(false);
}

template <typename TFromView, typename TToView>
void copyBySlices(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<3> )
{
	// static_assert(TFromView::kDimension == 3, "Sliced copy is only for 3D views (source fail)");
	// static_assert(TToView::kDimension == 3, "Sliced copy is only for 3D views (target fail)");
	// BOLT_DFORMAT("copy by slices:");
	auto src_pointer = from_view.pointer();
	auto dst_pointer = to_view.pointer();
	for (int i = 0; i < from_view.size()[2]; ++i) {
		cudaMemcpy3DParms parameters = { 0 };

		parameters.srcPtr = stridesToPitchedPtr(src_pointer, removeDimension(from_view.size(), 2), removeDimension(from_view.strides(), 2));
		parameters.dstPtr = stridesToPitchedPtr(dst_pointer, removeDimension(to_view.size(), 2), removeDimension(to_view.strides(), 2));
		parameters.extent = makeCudaExtent(sizeof(typename TFromView::Element), removeDimension(from_view.size(), 2));
		parameters.kind = kind;
		BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
		src_pointer += from_view.strides()[2];
		dst_pointer += to_view.strides()[2];
	}
}

// Specialization for 1D, so it can be used also for array views.
// Generic implementation can cause compile time error, due to the copyBySlices call.
template <typename TFromView, typename TToView>
void heterogeneousCopyAsync(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<1>)
{
	cudaMemcpy3DParms parameters = { 0 };

	parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.size(), from_view.strides());
	parameters.dstPtr = stridesToPitchedPtr(to_view.pointer(), to_view.size(), to_view.strides());
	parameters.extent = makeCudaExtent(sizeof(typename TFromView::Element), from_view.size());
	parameters.kind = kind;
	BOLT_DFORMAT("Copy pitched data: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstPtr, parameters.extent);
	BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
}

template <int tDimension, typename TFromView, typename TToView>
void heterogeneousCopyAsync(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<tDimension>)
{
	// Copy without padding
	// TODO(johny) - handle more complicated memory layouts.
	/*if (!from_view.HasContiguousMemory()) {
		BOLT_DFORMAT("Source device view does not have contiguous memory. Size: %1% Strides: %2%", from_view.size(), from_view.strides());
		BOLT_THROW(ContiguousMemoryNeeded());
	}
	if (!to_view.HasContiguousMemory()) {
		BOLT_DFORMAT("Target host view does not have contiguous memory. Size: %1% Strides: %2%", to_view.size(), to_view.strides());
		BOLT_THROW(ContiguousMemoryNeeded());
	}
	BOLT_CHECK(cudaMemcpyAsync(
			to_view.pointer(),
			from_view.pointer(),
			from_view.elementCount() * sizeof(typename TToView::Element),
			cudaMemcpyDeviceToHost,
			cuda_stream));*/
	if (!isContinuousInZ(from_view) || !isContinuousInZ(to_view)) {
		copyBySlices(from_view, to_view, kind, cuda_stream, DimensionValue<tDimension>());
	} else {
		cudaMemcpy3DParms parameters = { 0 };

		parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.size(), from_view.strides());
		parameters.dstPtr = stridesToPitchedPtr(to_view.pointer(), to_view.size(), to_view.strides());
		parameters.extent = makeCudaExtent(sizeof(typename TFromView::Element), from_view.size());
		parameters.kind = kind;
		BOLT_DFORMAT("Copy pitched data: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstPtr, parameters.extent);
		BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
	}
}


template <typename TFromView, typename TToView>
void copyDeviceToHostAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	using FromElement = typename std::remove_cv<typename TFromView::Element>::type;
	using ToElement = typename std::remove_cv<typename TToView::Element>::type;
	static_assert(std::is_same<FromElement, ToElement>::value, "From/To views have incompatible element types.");
	static_assert(IsMemcpyAble<TFromView>::value, "Source view must be memcpy able"); // NOTE(fidli): usually these asserts mean that used view transforms data and we cannot do that on the fly, make sure that you copy your data first (they get transformed, so that then memcpy can be called)
	static_assert(IsMemcpyAble<TToView>::value, "Target view must be memcpy able");

	heterogeneousCopyAsync(from_view, to_view, cudaMemcpyDeviceToHost, cuda_stream, DimensionValue<TFromView::kDimension>());
}


template <typename TFromView, typename TToView>
void copyHostToDeviceAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	using FromElement = typename std::remove_cv<typename TFromView::Element>::type;
	using ToElement = typename std::remove_cv<typename TToView::Element>::type;
	static_assert(std::is_same<FromElement, ToElement>::value, "From/To views have incompatible element types.");
	static_assert(IsMemcpyAble<TFromView>::value, "Source view must be memcpy able"); // NOTE(fidli): usually these asserts mean that used view transforms data and we cannot do that on the fly, make sure that you copy your data first (they get transformed, so that then memcpy can be called)
	static_assert(IsMemcpyAble<TToView>::value, "Target view must be memcpy able");

	heterogeneousCopyAsync(from_view, to_view, cudaMemcpyHostToDevice, cuda_stream, DimensionValue<TFromView::kDimension>());
}


template <typename TFromView, typename TToView, std::enable_if_t<IsImageView<TFromView>::value, int> = 0>
inline void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	DeviceToDeviceTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyDeviceToDeviceAsync(from_view, to_view, cuda_stream);
}


template <typename TFromView, typename TToView, std::enable_if_t<IsImageView<TFromView>::value, int> = 0>
inline void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	DeviceToHostTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyDeviceToHostAsync(from_view, to_view, cuda_stream);
}


template <typename TFromView, typename TToView, std::enable_if_t<IsImageView<TFromView>::value, int> = 0>
inline void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	HostToDeviceTag /*tag*/,
	cudaStream_t cuda_stream)
{
	copyHostToDeviceAsync(from_view, to_view, cuda_stream);
}
#endif  // __CUDACC__

template <typename TFromView, typename TToView, std::enable_if_t<IsImageView<TFromView>::value, int> = 0>
inline void asyncCopyHelper(
	TFromView from_view,
	TToView to_view,
	HostToHostTag /*tag*/,
	cudaStream_t /*unused*/)
{
	copyHostToHost(from_view, to_view);
}

template<bool tFromIsDeviceView, bool tFromIsHostView, bool tToIsDeviceView, bool tToIsHostView>
struct CopyDirection;

template<bool tFromIsDeviceView, bool tFromIsHostView>
struct CopyDirection<tFromIsDeviceView, tFromIsHostView, true, false> {
	using type = CopyDirectionTag<tFromIsDeviceView, true>;
};

template<bool tFromIsDeviceView, bool tFromIsHostView>
struct CopyDirection<tFromIsDeviceView, tFromIsHostView, false, true> {
	using type = CopyDirectionTag<!tFromIsHostView, false>;
};

// Use device copy for device input and hybrid output
template<bool tFromIsHostView>
struct CopyDirection<true, tFromIsHostView, true, true> {
	using type = CopyDirectionTag<true, true>;
};

// Use host copy for host input and hybrid output
template<bool tFromIsHostView>
struct CopyDirection<false, tFromIsHostView, true, true> {
	using type = CopyDirectionTag<false, false>;
};

// TODO(johny) implement special cases, unified memory, etc.



template <typename TFromView, typename TToView>
void copyAsync(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	// static_assert(IsImageView<TFromView>::value || IsArrayView<TFromView>::value, "Input must be a view");
	// static_assert(IsImageView<TToView>::value || IsArrayView<TToView>::value, "Output must be a view");
	// static_assert(TFromView::kDimension == TToView::kDimension, "Copy can be done only between image views of same dimensionality.");
	// // BOLT_DFORMAT("Copy sizes: \n  src: %1%\n  dst: %2%", from_view.size(), to_view.size());
	// if (from_view.size() != to_view.size()) {
	// 	BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(from_view.size(), to_view.size()));
	// }

	asyncCopyHelper(
		from_view,
		to_view,
		typename CopyDirection<TFromView::kIsDeviceView, TFromView::kIsHostView, TToView::kIsDeviceView, TToView::kIsHostView>::type(),
		cuda_stream);
}


template <typename TFromView, typename TToView>
void copy(
	TFromView from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	copyAsync(from_view, to_view, cuda_stream);

	// TODO(johny) - handle non-cuda code in better way
	#ifdef __CUDACC__
	if (TFromView::kIsDeviceView || TToView::kIsDeviceView) {
		BOLT_CHECK(cudaStreamSynchronize(cuda_stream));
	}
	#endif  // __CUDACC__
}


}  // namespace bolt
