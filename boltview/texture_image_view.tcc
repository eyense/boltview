// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomaskrupka@eyen.se

#pragma once

namespace bolt {

template <typename TToView, typename TElement, int tDimension, typename TCudaType>
void copyDeviceToHostAsync(
	TextureImageView<TElement, tDimension, TCudaType> from_view,
	TToView to_view,
	cudaStream_t cuda_stream)
{
	using ToElement = typename std::remove_cv<typename TToView::Element>::type;
	static_assert(std::is_same<TElement, ToElement>::value, "From/To views have incompatible element types.");
	static_assert(TToView::kIsMemoryBased, "Target view must be memory based");

	if(!isContinuousInZ(from_view)) {
		BOLT_THROW(ContiguousMemoryNeeded());
	}

	cudaMemcpy3DParms parameters = { 0 };

	parameters.srcArray = from_view.array();
	parameters.dstPtr = stridesToPitchedPtr(to_view.pointer(), to_view.size(), to_view.strides());
	parameters.extent = makeCudaExtent(1, from_view.size());
	parameters.kind = cudaMemcpyDeviceToHost;
	BOLT_DFORMAT("Copy data from texture: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcArray, parameters.dstPtr, parameters.extent);
	BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
}

template <typename TFromView, typename TToView>
void textureCopyBySlices(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<2>,
	Int2 offset)
{
	BOLT_ASSERT(false);
	BOLT_THROW(NotYetImplemented());
}

/*
template<typename TView>
void FillSourceMemcpyParameters(TView view, cudaMemcpy3DParms &parameters) {
	parameters.srcPtr = stridesToPitchedPtr(view.pointer(), view.size(), view.strides());
	parameters.srcPos = makeCudaPos(Int3());
	parameters.extent = makeCudaExtent(1, view.size());
}

template<typename TView>
void FillDestinationMemcpyParameters(TView view, cudaMemcpy3DParms &parameters) {
	parameters.dstArray = view.array();
	parameters.dstPos = makeCudaPos(to_view.corner());
}

template <typename TElement, int tDimension, typename TCudaType>
void FillDestinationMemcpyParameters(TextureImageView<TElement, tDimension, TCudaType> view, cudaMemcpy3DParms &parameters) {
	parameters.dstArray = view.array();
	parameters.dstPos = makeCudaPos(view.corner());
}
*/

template <typename TFromView, typename TToView>
void textureCopyBySlices(
	TFromView from_view,
	TToView to_view,
	cudaMemcpyKind kind,
	cudaStream_t cuda_stream,
	DimensionValue<3>,
	Int3 offset)
{
	auto src_pointer = from_view.pointer();
	Int3 dst_pos = offset;
	BOLT_DFORMAT("Copy data to texture by slices");
	for (int i = 0; i < from_view.size()[2]; ++i) {
		cudaMemcpy3DParms parameters = { 0 };

		parameters.srcPtr = stridesToPitchedPtr(src_pointer, removeDimension(from_view.size(), 2), removeDimension(from_view.strides(), 2));
		parameters.dstArray = to_view.array();
		parameters.dstPos = makeCudaPos(to_view.corner() + Int3(0, 0, i));
		parameters.extent = makeCudaExtent(1, removeDimension(from_view.size(), 2));
		parameters.kind = kind;
		BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
		src_pointer += from_view.strides()[2];
		dst_pos[2]++;
	}
}

template <typename TFromView, typename TElement, int tDimension, typename TCudaType>
struct CopyHostToDeviceAsync {
	static void call(
		TFromView from_view,
		TextureImageView<TElement, tDimension, TCudaType> to_view,
		cudaStream_t cuda_stream)
	{

		if(!isContinuousInZ(from_view)){
			BOLT_DFORMAT("Copy data to texture by slices");
			textureCopyBySlices(from_view, to_view, cudaMemcpyHostToDevice, cuda_stream, DimensionValue<TFromView::kDimension>(), to_view.corner());
		} else {
			cudaMemcpy3DParms parameters = { 0 };
			parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.size(), from_view.strides());
			parameters.dstArray = to_view.array();
			parameters.dstPos = makeCudaPos(to_view.corner());
			parameters.extent = makeCudaExtent(1, from_view.size());
			parameters.kind = cudaMemcpyHostToDevice;

			BOLT_DFORMAT("Copy data to texture: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstArray, parameters.extent);
			BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
		}
	}

};

template <typename TFromView, typename TElement, typename TCudaType>
struct CopyHostToDeviceAsync<TFromView, TElement, 1, TCudaType> {
	static void call(
		TFromView from_view,
		TextureImageView<TElement, 1, TCudaType> to_view,
		cudaStream_t cuda_stream)
	{
			cudaMemcpy3DParms parameters = { 0 };
			parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.size(), from_view.strides());
			parameters.dstArray = to_view.array();
			parameters.dstPos = makeCudaPos(to_view.corner());
			parameters.extent = makeCudaExtent(1, from_view.size());
			parameters.kind = cudaMemcpyHostToDevice;

			BOLT_DFORMAT("Copy data to texture: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstArray, parameters.extent);
			BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
	}

};

template <typename TFromView, typename TElement, int tDimension, typename TCudaType>
void copyHostToDeviceAsync(
	TFromView from_view,
	TextureImageView<TElement, tDimension, TCudaType> to_view,
	cudaStream_t cuda_stream)
{
	using FromElement = typename std::remove_cv<typename TFromView::Element>::type;
	static_assert(std::is_same<FromElement, TElement>::value, "From/To views have incompatible element types.");
	static_assert(TFromView::kIsMemoryBased, "Source view must be memory based");

	CopyHostToDeviceAsync<TFromView, TElement, tDimension, TCudaType>::call(from_view, to_view, cuda_stream);
}

template <typename TFromView, typename TElement, int tDimension, typename TCudaType>
void copyDeviceToDeviceAsync(
	TFromView from_view,
	TextureImageView<TElement, tDimension, TCudaType> to_view,
	cudaStream_t cuda_stream)
{
	using FromElement = typename std::remove_cv<typename TFromView::Element>::type;
	static_assert(std::is_same<FromElement, TElement>::value, "From/To views have incompatible element types.");
	static_assert(TFromView::kIsMemoryBased, "Source view must be memory based");

	cudaMemcpy3DParms parameters = { 0 };

	Int3 dst_pos;
	if(!isContinuousInZ(from_view)){
		BOLT_DFORMAT("Copy data to texture by slices");
		textureCopyBySlices(from_view, to_view, cudaMemcpyDeviceToDevice, cuda_stream, DimensionValue<TFromView::kDimension>(), to_view.corner());
	} else {
		parameters.srcPtr = stridesToPitchedPtr(from_view.pointer(), from_view.size(), from_view.strides());
		parameters.dstArray = to_view.array();
		parameters.dstPos = makeCudaPos(to_view.corner());
		parameters.extent = makeCudaExtent(1, from_view.size());
		parameters.kind = cudaMemcpyDeviceToDevice;

		BOLT_DFORMAT("Copy data to texture: \n  src: %1%\n  dst: %2%\n  extent: %3%", parameters.srcPtr, parameters.dstArray, parameters.extent);
		BOLT_CHECK(cudaMemcpy3DAsync(&parameters, cuda_stream));
	}
}


}  // namespace bolt
