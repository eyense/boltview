// Copyright 2015-20 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <algorithm>
#include <complex>
#include <map>
#include <string>
#include <utility>
#include <boost/format.hpp>

#if defined(__CUDACC__)
#include <boltview/cuda_defines.h>
#include <boltview/cuda_utils.h>
#include <boltview/device_image_view.h>
#endif  // defined(__CUDACC__)

#include <boltview/procedural_views.h>
#include <boltview/transform.h>

#include <boltview/math/complex.h>

namespace bolt {

/// \addtogroup FFT
/// @{

/// Returns image size of individual images, if no stack dimension is specified, returns size
template<typename TStackPolicy = Stack<-1, -1>, int tDim>
Vector<int, tDim - TStackPolicy::kIsStack> subSize(Vector<int, tDim> size){
	Vector<int, tDim - TStackPolicy::kIsStack> result;
	int write_i = 0;
	for(int i = 0; i < tDim; i++){
		if(i != TStackPolicy::kDimension1 && i != TStackPolicy::kDimension2){
			set(result, write_i, get(size, i));
			write_i++;
		}
	}
	return result;
}

/// Returns image stack size of individual images, if no stack dimension is specified, returns 1
template<typename TStackPolicy = Stack<-1, -1>, int tDim>
int stackSize(Vector<int, tDim> size){
	int result = 1;
	int write_i = 0;
	for(int i = 0; i < tDim; i++){
		if(i == TStackPolicy::kDimension1 || i == TStackPolicy::kDimension2){
			result *= get(size, i);
			write_i++;
		}
	}
	return result;
}

/// Returns jump  size to next sample element within one image
template<typename TStackPolicy = Stack<-1, -1>, int tDim>
int nextElementHop(Vector<int, tDim> strides){
	// assuming z y x layout
	if (!TStackPolicy::kIsStack) {
		return 1;
	}
	if (TStackPolicy::kDimension2 == -1) {
		return strides[(TStackPolicy::kDimension1 == 0 ? 1 : 0)];
	}
	// else
	int dim = 3 - TStackPolicy::kDimension1 - TStackPolicy::kDimension2;
	return strides[dim];
}

/// Returns jump  size to next sample image
template<typename TStackPolicy = Stack<-1, -1>, int tDim>
int nextInputHop(Vector<int, tDim> size, Vector<int, tDim> strides){
	// assuming z y x layout
	if (!TStackPolicy::kIsStack) {
		return product(size);
	}
	if (TStackPolicy::kDimension2 == -1) {
		return strides[TStackPolicy::kDimension1];
	}
	static_assert(TStackPolicy::kDimension2 == -1 || (TStackPolicy::kDimension1 == 1 || TStackPolicy::kDimension2 == 1), "Cannot have data in Y (1) dimension, its impossible with 2D stack of 1D images in ZYX memory layout");
	// data are in X dimension
	if (TStackPolicy::kDimension1 != 0 && TStackPolicy::kDimension2 != 0){
		return strides[1];
	}
	// else, data are in Z dimension
	return strides[0];
}
///
/// \brief Internal helper function for libraries plan
///
/// NOTE(fidli): these are cufft/fftw specific, might make sense for more libraries
template<typename TStackPolicy>
struct EmbeddingFromStridesAndSizesHelper{

	template<int tStrideDim, int tSizeDim>
	static inline Vector<int, tStrideDim> get(Vector<int, tStrideDim> strides, Vector<int, tSizeDim> sizes);

	static inline Vector<int, 1> get(Vector<int, 2> strides, Vector<int, 1>  /*sizes*/){
		static_assert(TStackPolicy::kDimension1 != -1 && TStackPolicy::kDimension1 >= 0 && TStackPolicy::kDimension1 <= 1, "Must be stack or stack and in existing dimension");
		static_assert(TStackPolicy::kDimension2 == -1, "Cannot be double stack");
		// assuming X,Y*width,Z*width*height data layout
		// TODO(fidli): NON X,Y,Z layout
		Vector<int, 1> result(strides[1-TStackPolicy::kDimension1]);
		return result;
	}

	static inline Vector<int, 2> get(Vector<int, 3> strides, Vector<int, 2>  /*sizes*/){
		static_assert(TStackPolicy::kDimension1 != -1 && TStackPolicy::kDimension1 >= 0 && TStackPolicy::kDimension1 <= 2, "Must be stack or stack and in existing dimension");
		static_assert(TStackPolicy::kDimension2 == -1, "Cannot be double stack");
		// TODO(fidli): consider strides, now we assume data are packed assuming X,Y*width,Z*width*height data layout
		// TODO(fidli): NON X,Y,Z layout
		int stack_stride_index = TStackPolicy::kDimension1 == 2 ? 1 : 2;
		int lower_stack_stride = stack_stride_index-1;
		if(lower_stack_stride == TStackPolicy::kDimension1){
			lower_stack_stride--;
		}
		Vector<int, 2> result(strides[TStackPolicy::kDimension1 == 0 ? 1 : 0], strides[stack_stride_index]/strides[lower_stack_stride]);
		return result;
	}

	static inline Vector<int, 1> get(Vector<int, 3> strides, Vector<int, 1>  /*sizes*/){
		static_assert(TStackPolicy::kDimension1 != -1 && TStackPolicy::kDimension1 >= 0 && TStackPolicy::kDimension1 <= 2, "Must be stack or stack and in existing dimension");
		static_assert(TStackPolicy::kDimension2 != -1 && TStackPolicy::kDimension2 >= 0 && TStackPolicy::kDimension1 <= 2 && TStackPolicy::kDimension1 != TStackPolicy::kDimension2, "Must be dual stack");
		// TODO(fidli): consider strides, now we assume data are packed assuming X,Y*width,Z*width*height data layout
		// TODO(fidli): NON X,Y,Z layout
		int dim = 3 - TStackPolicy::kDimension1 - TStackPolicy::kDimension2;
		Vector<int, 1> result(strides[dim]);
		return result;
	}
};

template<>
struct EmbeddingFromStridesAndSizesHelper<Stack<-1, -1>>{
	template<int tDim>
	static inline Vector<int, tDim> get(Vector<int, tDim>  /*strides*/, Vector<int, tDim> sizes){
		// TODO(fidli): consider strides, now we assume data are packed assuming X,Y*width,Z*width*height data layout
		// TODO(fidli): NON X,Y,Z layout
		return sizes;
	}
};

// TODO(fidli): cummulate these, when compiler wont complain about parametrized innerts
template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<-1, -1>>(Int3 size) {
	return Int3(size[0] / 2 + 1, size[1], size[2]);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<1, -1>>(Int3 size) {
	return Int3(size[0] / 2 + 1, size[1], size[2]);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<2, -1>>(Int3 size) {
	return Int3(size[0] / 2 + 1, size[1], size[2]);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<0, -1>>(Int3 size) {
	return Int3(size[0], size[1]  / 2 + 1, size[2]);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<0, 1>>(Int3 size) {
	return Int3(size[0], size[1], size[2] / 2 + 1);

}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<1, 0>>(Int3 size) {
	return Int3(size[0], size[1], size[2] / 2 + 1);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<0, 2>>(Int3 size) {
	return Int3(size[0], size[1] / 2 + 1, size[2]);

}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<2, 0>>(Int3 size) {
	return Int3(size[0], size[1] / 2 + 1, size[2]);

}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<1, 2>>(Int3 size) {
	return Int3(size[0] / 2 + 1, size[1], size[2]);
}

template<>
BOLT_DECL_HYBRID inline Int3 getFftImageSize<Stack<2, 1>>(Int3 size) {
	return Int3(size[0] / 2 + 1, size[1], size[2]);
}

template<typename TStackPolicy>
BOLT_DECL_HYBRID inline Int2 getFftImageSize(Int2 size){
	static_assert(TStackPolicy::kDimension1 != 0, "Bad function");
	static_assert(TStackPolicy::kDimension1 <= 1, "Can stack only or X, or Y");
	static_assert(TStackPolicy::kDimension2 == -1, "Cannot 2D stack when using 2D input");
	return Int2(size[0] / 2 + 1, size[1]);
};

template<>
BOLT_DECL_HYBRID inline Int2 getFftImageSize<Stack<0, -1>>(Int2 size){
	return Int2(size[0], size[1] / 2 + 1);
}

template<typename TStackPolicy>
BOLT_DECL_HYBRID inline Vector<int,1> getFftImageSize(int size) {
	static_assert(TStackPolicy::kDimension1 == -1 && TStackPolicy::kDimension2 == -1, "No stacking allowed");
	return Vector<int,1>(size / 2 + 1);
}

template<typename TStackPolicy>
BOLT_DECL_HYBRID inline Vector<int,1> getFftImageSize(Vector<int,1> size) {
	static_assert(TStackPolicy::kDimension1 == -1 && TStackPolicy::kDimension2 == -1, "No stacking allowed");
	return Vector<int,1>(size[0] / 2 + 1);
}

template<typename TSizeType>
TSizeType reverseCoordinates(TSizeType size_vector){
	TSizeType result = size_vector;
	int lower_half_count = TSizeType::kDimension/2;
	int last_index = TSizeType::kDimension-1;
	for(int i = 0; i < lower_half_count; i++){
		std::swap(result[i], result[last_index-i]);
	}
	return result;
}


template<>
template<typename TSizeType, typename TStackPolicy>
std::unique_ptr<HostFftPlan<TSizeType>> HostFftPolicyHelper<Forward>::createPlan(TSizeType space_domain_size, TSizeType frequency_domain_size, TStackPolicy /*unused*/){
	auto plan = std::unique_ptr<HostFftPlan<TSizeType>>(new HostFftPlan<TSizeType>());
	if(plan.get() == nullptr){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Memory for input or output could not be allocated"));
	}

	BOLT_DFORMAT("FFT plans prepared. Size = %1%, Spectrum size = %2%", space_domain_size, frequency_domain_size);

	plan->inputSize = space_domain_size;
	plan->outputSize = frequency_domain_size;

	// FFTW has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto space_domain_sub_size = subSize<TStackPolicy>(space_domain_size);
	auto space_domain_sub_size_r = reverseCoordinates(space_domain_sub_size);
	auto frequency_domain_sub_size = subSize<TStackPolicy>(frequency_domain_size);
	auto frequency_domain_sub_size_r = reverseCoordinates(frequency_domain_sub_size);

	auto space_domain_strides = stridesFromSize(space_domain_size);
	auto frequency_domain_strides = stridesFromSize(frequency_domain_size);
	auto space_domain_strides_r = reverseCoordinates(space_domain_strides);
	auto frequency_domain_strides_r = reverseCoordinates(frequency_domain_strides);


	plan->spaceDomainData = std::unique_ptr<float[], typename HostFftPlan<TSizeType>::FFTWDataDestroyer>((float *) fftwf_malloc(sizeof(float) * product(space_domain_size)));
	plan->frequencyDomainData = std::unique_ptr<fftwf_complex[], typename HostFftPlan<TSizeType>::FFTWDataDestroyer>((fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * product(frequency_domain_size)));

	if(plan->spaceDomainData.get() == NULL || plan->frequencyDomainData.get() == NULL){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Memory for input or output could not be allocated"));
	}

	/* This is same for cufft and libfftw:
	This can be found here: https://docs.nvidia.com/cuda/cufft/index.html
	under 2.6 advanced data layout (cufftPlanMany)

	b is stack index
	dist is length between stack elements
	stride is advance in lowest dimension - typically X (1)
	onembed are advances in other dimensions

	1D
	input[ b * idist + x * istride]
	output[ b * odist + x * ostride]

	2D
	input[b * idist + (x * inembed[1] + y) * istride]
	output[b * odist + (x * onembed[1] + y) * ostride]

	3D
	input[b * idist + ((x * inembed[1] + y) * inembed[2] + z) * istride]
	output[b * odist + ((x * onembed[1] + y) * onembed[2] + z) * ostride]
	*/

	*(plan.get()->plan.get()) = fftwf_plan_many_dft_r2c(
			TSizeType::kDimension - TStackPolicy::kIsStack,	 // rank(dimension)
			space_domain_sub_size_r.pointer(),			  // n - size of each dimension, this is always in space domain
			stackSize<TStackPolicy>(space_domain_size), // stack size
			plan.get()->spaceDomainData.get(),  // input array
			EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(space_domain_strides, space_domain_sub_size_r).pointer(), // inembed
			nextElementHop<TStackPolicy>(space_domain_strides), // istride, jump to next element
			nextInputHop<TStackPolicy>(space_domain_size, space_domain_strides), // idist jump to next input data
			plan.get()->frequencyDomainData.get(), // output array
			EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequency_domain_strides, frequency_domain_sub_size_r).pointer(), // onenbed
			nextElementHop<TStackPolicy>(frequency_domain_strides), // ostride jump to next sample
			nextInputHop<TStackPolicy>(frequency_domain_size, frequency_domain_strides), // odist jump to next input data
			FFTW_MEASURE// flags
			);

	if(plan.get()->plan.get() == NULL){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Plan could not be created"));
	}
	plan->frequencyDomainView = makeHostImageView((HostComplexType *)(plan.get()->frequencyDomainData.get()), frequency_domain_size);
	plan->spaceDomainView = makeHostImageView(plan.get()->spaceDomainData.get(), space_domain_size);
	return plan;
}

template<>
template<typename TSizeType, typename TStackPolicy>
std::unique_ptr<HostFftPlan<TSizeType>> HostFftPolicyHelper<Inverse>::createPlan(TSizeType space_domain_size, TSizeType frequency_domain_size, TStackPolicy /*unused*/){
	auto plan = std::unique_ptr<HostFftPlan<TSizeType>>(new HostFftPlan<TSizeType>());
	if(plan.get() == nullptr){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Memory for input or output could not be allocated"));
	}

	BOLT_DFORMAT("FFT plans prepared. Size = %1%, Spectrum size = %2%", space_domain_size, frequency_domain_size);

	plan->inputSize = frequency_domain_size;
	plan->outputSize = space_domain_size;

	// FFTW has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto space_domain_sub_size = subSize<TStackPolicy>(space_domain_size);
	auto space_domain_sub_size_r = reverseCoordinates(space_domain_sub_size);
	auto frequency_domain_sub_size = subSize<TStackPolicy>(frequency_domain_size);
	auto frequency_domain_sub_size_r = reverseCoordinates(frequency_domain_sub_size);

	auto space_domain_strides = stridesFromSize(space_domain_size);
	auto frequency_domain_strides = stridesFromSize(frequency_domain_size);
	auto space_domain_strides_r = reverseCoordinates(space_domain_strides);
	auto frequency_domain_strides_r = reverseCoordinates(frequency_domain_strides);

	plan->spaceDomainData = std::unique_ptr<float[], typename HostFftPlan<TSizeType>::FFTWDataDestroyer>((float *) fftwf_malloc(sizeof(float) * product(space_domain_size)));
	plan->frequencyDomainData = std::unique_ptr<fftwf_complex[], typename HostFftPlan<TSizeType>::FFTWDataDestroyer>((fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * product(frequency_domain_size)));

	if(plan->spaceDomainData.get() == NULL || plan->frequencyDomainData.get() == NULL){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Memory for input or output could not be allocated"));
	}

	*(plan.get()->plan.get()) = fftwf_plan_many_dft_c2r(
			TSizeType::kDimension - TStackPolicy::kIsStack,								 // rank(dimension)
			space_domain_sub_size_r.pointer(),			  // n - size of each dimension, this is always in space domain
			stackSize<TStackPolicy>(frequency_domain_size), // stack size
			plan.get()->frequencyDomainData.get(),  // input array
			EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequency_domain_strides, frequency_domain_sub_size_r).pointer(), // inenbed
			nextElementHop<TStackPolicy>(frequency_domain_strides), // istride jump to next sample
			nextInputHop<TStackPolicy>(frequency_domain_size, frequency_domain_strides), // idist jump to next input data
			plan.get()->spaceDomainData.get(), // output array
			EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(space_domain_strides, space_domain_sub_size_r).pointer(), // onenbed
			nextElementHop<TStackPolicy>(space_domain_strides), // ostride jump to next sample
			nextInputHop<TStackPolicy>(space_domain_size, space_domain_strides), // odist jump to next input data
			FFTW_MEASURE// flags
	);

	if(plan.get()->plan.get() == NULL){
		BOLT_THROW(FFTWError() << bolt::MessageErrorInfo("Plan could not be created"));
	}
	plan->frequencyDomainView = makeHostImageView((HostComplexType *)(plan.get()->frequencyDomainData.get()), frequency_domain_size);
	plan->spaceDomainView = makeHostImageView(plan.get()->spaceDomainData.get(), space_domain_size);
	return plan;
}

template<typename TDirection>
template<typename TSizeType, typename TInputView, typename TOutputView>
void HostFftPolicyHelper<TDirection>::requireDimensions(const HostFftPlan<TSizeType> * plan, const TInputView input, const TOutputView output){
	if (plan->inputSize != input.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(plan->inputSize, input.size()));
	}
	if (plan->outputSize != output.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(plan->outputSize, output.size()));
	}
}


template<>
template<typename TSizeType, typename TInputView, typename TOutputView>
void HostFftPolicyHelper<Forward>::calculate(const HostFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
	static_assert(IsHostView<TInputView>::value && IsHostView<TOutputView>::value, "Both views must be host views");
	BOLT_DFORMAT("FFT Forward transformation: size %1%", input.size());
	requireDimensions(plan, input, output);
	static_assert(sizeof(fftwf_complex) == sizeof(typename TOutputView::Element), "Output view element size must match size of fftwf_complex");
	copy(input, plan->spaceDomainView);
	fftwf_execute(*(plan->plan.get()));
	copy(plan->frequencyDomainView, output);
}

template<>
template<typename TSizeType, typename TInputView, typename TOutputView>
void HostFftPolicyHelper<Inverse>::calculate(const HostFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
	static_assert(IsHostView<TInputView>::value && IsHostView<TOutputView>::value, "Both views must be host views");
	BOLT_DFORMAT("FFT Forward transformation: size %1%", input.size());
	requireDimensions(plan, input, output);
	static_assert(sizeof(float) == sizeof(typename TOutputView::Element), "Output view element size must match size of float");
	copy(input, plan->frequencyDomainView);
	fftwf_execute(*(plan->plan.get()));
	copy(plan->spaceDomainView, output);
}

#ifdef __CUDACC__

template<>
template<typename TSizeType,typename TStackPolicy>
uint64_t DeviceFftPolicyHelper<Forward>::estimateWorkArea(TSizeType spaceDomainSize, TSizeType frequencyDomainSize, TStackPolicy){
	// CUFFT has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto spaceDomainSubSize = subSize<TStackPolicy>(spaceDomainSize);
	auto spaceDomainSubSizeR = reverseCoordinates(spaceDomainSubSize);
	auto frequencyDomainSubSize = subSize<TStackPolicy>(frequencyDomainSize);
	auto frequencyDomainSubSizeR = reverseCoordinates(frequencyDomainSubSize);

	auto spaceDomainStrides = stridesFromSize(spaceDomainSize);
	auto frequencyDomainStrides = stridesFromSize(frequencyDomainSize);
	auto spaceDomainStridesR = reverseCoordinates(spaceDomainStrides);
	auto frequencyDomainStridesR = reverseCoordinates(frequencyDomainStrides);

	size_t result;
	BOLT_CUFFT_CHECK(cufftEstimateMany(
					TSizeType::kDimension - TStackPolicy::kIsStack,		 // rank(dimension)
					spaceDomainSubSizeR.pointer(),			  // n - size of each dimension, this is always in space domain
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(spaceDomainStrides, spaceDomainSubSizeR).pointer(), // inembed
					nextElementHop<TStackPolicy>(spaceDomainStrides), // istride jump to next sample
					nextInputHop<TStackPolicy>(spaceDomainSize, spaceDomainStrides), // idist jump to next input data
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequencyDomainStrides, frequencyDomainSubSizeR).pointer(), // onenbed
					nextElementHop<TStackPolicy>(frequencyDomainStrides), // ostride jump to next sample
					nextInputHop<TStackPolicy>(frequencyDomainSize, frequencyDomainStrides), // odist jump to next input data
					CUFFT_R2C,						 // type
					stackSize<TStackPolicy>(spaceDomainSize), // stack size
					&result));
	return static_cast<uint64_t>(result);
}

template<>
template<typename TSizeType, typename TStackPolicy>
uint64_t DeviceFftPolicyHelper<Inverse>::estimateWorkArea(TSizeType spaceDomainSize, TSizeType frequencyDomainSize, TStackPolicy){
	// CUFFT has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto spaceDomainSubSize = subSize<TStackPolicy>(spaceDomainSize);
	auto spaceDomainSubSizeR = reverseCoordinates(spaceDomainSubSize);
	auto frequencyDomainSubSize = subSize<TStackPolicy>(frequencyDomainSize);
	auto frequencyDomainSubSizeR = reverseCoordinates(frequencyDomainSubSize);

	auto spaceDomainStrides = stridesFromSize(spaceDomainSize);
	auto frequencyDomainStrides = stridesFromSize(frequencyDomainSize);
	auto spaceDomainStridesR = reverseCoordinates(spaceDomainStrides);
	auto frequencyDomainStridesR = reverseCoordinates(frequencyDomainStrides);

	size_t result;
	BOLT_CUFFT_CHECK(cufftEstimateMany(
					TSizeType::kDimension - TStackPolicy::kIsStack,								 // rank(dimension)
					spaceDomainSubSizeR.pointer(),			  // n - size of each dimension, this is always in space domain
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequencyDomainStrides, frequencyDomainSubSizeR).pointer(), // inenbed
					nextElementHop<TStackPolicy>(frequencyDomainStrides), // istride jump to next sample
					nextInputHop<TStackPolicy>(frequencyDomainSize, frequencyDomainStrides), // idist jump to next input data
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(spaceDomainStrides, spaceDomainSubSizeR).pointer(), // onembed
					nextElementHop<TStackPolicy>(spaceDomainStrides), // ostride jump to next sample
					nextInputHop<TStackPolicy>(spaceDomainSize, spaceDomainStrides), // odist jump to next input data
					CUFFT_C2R,						 // type
					stackSize<TStackPolicy>(frequencyDomainSize), // stack size
					&result));
	return static_cast<uint64_t>(result);
}

template<>
template<typename TSizeType, typename TStackPolicy>
std::unique_ptr<DeviceFftPlan<TSizeType>> DeviceFftPolicyHelper<Forward>::createPlan(TSizeType spaceDomainSize, TSizeType frequencyDomainSize, TStackPolicy){
	auto plan = std::unique_ptr<DeviceFftPlan<TSizeType>>(new DeviceFftPlan<TSizeType>());
	BOLT_CUFFT_CHECK(cufftCreate(plan.get()->plan.get()));
	BOLT_DFORMAT("FFT plans prepared. Size = %1%, Spectrum size = %2%", spaceDomainSize, frequencyDomainSize);

	plan->inputSize = spaceDomainSize;
	plan->outputSize = frequencyDomainSize;

	// CUFFT has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto spaceDomainSubSize = subSize<TStackPolicy>(spaceDomainSize);
	auto spaceDomainSubSizeR = reverseCoordinates(spaceDomainSubSize);
	auto frequencyDomainSubSize = subSize<TStackPolicy>(frequencyDomainSize);
	auto frequencyDomainSubSizeR = reverseCoordinates(frequencyDomainSubSize);

	auto spaceDomainStrides = stridesFromSize(spaceDomainSize);
	auto frequencyDomainStrides = stridesFromSize(frequencyDomainSize);
	auto spaceDomainStridesR = reverseCoordinates(spaceDomainStrides);
	auto frequencyDomainStridesR = reverseCoordinates(frequencyDomainStrides);

	BOLT_CUFFT_CHECK(cufftPlanMany(
		plan.get()->plan.get(),  // plan
		TSizeType::kDimension - TStackPolicy::kIsStack,  // rank(dimension)
		spaceDomainSubSizeR.pointer(),  // n - size of each dimension, this is always in space domain
		EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(spaceDomainStrides, spaceDomainSubSizeR).pointer(), // inembed
		nextElementHop<TStackPolicy>(spaceDomainStrides),  // istride jump to next sample
		nextInputHop<TStackPolicy>(spaceDomainSize, spaceDomainStrides),  // idist jump to next input data
		EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequencyDomainStrides, frequencyDomainSubSizeR).pointer(),  // onenbed
		nextElementHop<TStackPolicy>(frequencyDomainStrides),  // ostride jump to next sample
		nextInputHop<TStackPolicy>(frequencyDomainSize, frequencyDomainStrides),  // odist jump to next input data
		CUFFT_R2C,  // type
		stackSize<TStackPolicy>(spaceDomainSize) // stack size
		));

	// Do we want this by default?
	// Put the FFT computation on the thread's stream. Otherwise, the default is the default stream, which is fine for single threaded apps
	cufftSetStream(*(plan.get()->plan.get()), CU_STREAM_PER_THREAD);

	return plan;
}

template<>
template<typename TSizeType, typename TStackPolicy>
std::unique_ptr<DeviceFftPlan<TSizeType>> DeviceFftPolicyHelper<Inverse>::createPlan(TSizeType spaceDomainSize, TSizeType frequencyDomainSize, TStackPolicy){
	auto plan = std::unique_ptr<DeviceFftPlan<TSizeType>>(new DeviceFftPlan<TSizeType>());
	BOLT_CUFFT_CHECK(cufftCreate(plan.get()->plan.get()));
	BOLT_DFORMAT("FFT plans prepared. Size = %1%, Spectrum size = %2%", spaceDomainSize, frequencyDomainSize);

	plan->inputSize = frequencyDomainSize;
	plan->outputSize = spaceDomainSize;


	// CUFFT has reversed coordinates - this was cause of the strange bug we had
	// This seems only for planning purpouses, and only for size parameter
	auto spaceDomainSubSize = subSize<TStackPolicy>(spaceDomainSize);
	auto spaceDomainSubSizeR = reverseCoordinates(spaceDomainSubSize);
	auto frequencyDomainSubSize = subSize<TStackPolicy>(frequencyDomainSize);
	auto frequencyDomainSubSizeR = reverseCoordinates(frequencyDomainSubSize);

	auto spaceDomainStrides = stridesFromSize(spaceDomainSize);
	auto frequencyDomainStrides = stridesFromSize(frequencyDomainSize);
	auto spaceDomainStridesR = reverseCoordinates(spaceDomainStrides);
	auto frequencyDomainStridesR = reverseCoordinates(frequencyDomainStrides);

	BOLT_CUFFT_CHECK(cufftPlanMany(
					plan.get()->plan.get(),					// plan
					TSizeType::kDimension - TStackPolicy::kIsStack,								 // rank(dimension)
					spaceDomainSubSizeR.pointer(),			  // n - size of each dimension, this is always in space domain
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(frequencyDomainStrides, frequencyDomainSubSizeR).pointer(), // inenbed
					nextElementHop<TStackPolicy>(frequencyDomainStrides), // istride jump to next sample
					nextInputHop<TStackPolicy>(frequencyDomainSize, frequencyDomainStrides), // idist jump to next input data
					EmbeddingFromStridesAndSizesHelper<TStackPolicy>::get(spaceDomainStrides, spaceDomainSubSizeR).pointer(), // onembed
					nextElementHop<TStackPolicy>(spaceDomainStrides), // ostride jump to next sample
					nextInputHop<TStackPolicy>(spaceDomainSize, spaceDomainStrides), // odist jump to next input data
					CUFFT_C2R,						 // type
					stackSize<TStackPolicy>(frequencyDomainSize) // stack size
					));

	// Do we want this by default?
	// Put the FFT computation on the thread's stream. Otherwise, the default is the default stream, which is fine for single threaded apps
	cufftSetStream(*(plan.get()->plan.get()), CU_STREAM_PER_THREAD);

	return plan;
}


template<typename TDirection>
template<typename TSizeType, typename TInputView, typename TOutputView>
void DeviceFftPolicyHelper<TDirection>::requireDimensions(const DeviceFftPlan<TSizeType> * plan, const TInputView input, const TOutputView output){
	if (plan->inputSize != input.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(plan->inputSize, input.size()));
	}
	if (plan->outputSize != output.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(plan->outputSize, output.size()));
	}
}

template<>
template<typename TSizeType, typename TInputView, typename TOutputView>
void DeviceFftPolicyHelper<Forward>::calculateAsync(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
	static_assert(IsDeviceView<TInputView>::value && IsDeviceView<TOutputView>::value, "Both views must be device views");
	BOLT_DFORMAT("FFT Forward transformation: size %1%", input.size());
	requireDimensions(plan, input, output);
	BOLT_CHECK_ERROR_STATE("Problem before forward FFT call");
	static_assert(sizeof(DeviceComplexType) == sizeof(typename TOutputView::Element), "Output view element size must match size of DeviceComplexType");
	BOLT_CUFFT_CHECK(cufftExecR2C(*(plan->plan.get()), input.pointer(), output.pointer()));
}

template<>
template<typename TSizeType, typename TInputView, typename TOutputView>
void DeviceFftPolicyHelper<Inverse>::calculateAsync(const DeviceFftPlan<TSizeType> * plan, TInputView input, TOutputView output){
	static_assert(IsDeviceView<TInputView>::value && IsDeviceView<TOutputView>::value, "Both views must be device views");
	BOLT_DFORMAT("FFT Inverse transformation: size %1%", input.size());
	requireDimensions(plan, input, output);
	BOLT_CHECK_ERROR_STATE("Problem before forward FFT call");
	static_assert(sizeof(DeviceComplexType) == sizeof(typename TInputView::Element), "Input view element size must match size of DeviceComplexType");
	BOLT_CUFFT_CHECK(cufftExecC2R(*(plan->plan.get()), input.pointer(), output.pointer()));
}

#endif  // __CUDACC__

/// @}

}  // namespace bolt
