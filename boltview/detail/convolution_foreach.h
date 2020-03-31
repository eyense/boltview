// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#pragma once

#include <boltview/math/vector.h>
#include <boltview/image_locator.h>
#include <boltview/loop_utils.h>
#include <boltview/transform.h>
#include <boltview/for_each.h>
#include <boltview/convolution_kernels.h>
#include <boltview/convolution.h>


namespace bolt {

/// Convolves view with convolution kernel
/// Used mainly for testing ForeachLocator
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param convolution_kernel Convolution kernel to be applied
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename TConvolutionKernel, typename TPolicy>
void convolutionForeach(TInView view_in, TOutView view_out, const TConvolutionKernel & convolution_kernel, TPolicy policy, cudaStream_t cuda_stream = 0){
	const bool views_host =  TInView::kIsHostView && TOutView::kIsHostView;
	const bool views_device =  TInView::kIsDeviceView && TOutView::kIsDeviceView;

	static_assert(views_host || views_device,
		"Incompatible views. Views have to be both device, both host or at least one of them has to be UnifiedImageView");

	const bool kernel_host_only = TConvolutionKernel::kIsHostKernel && !TConvolutionKernel::kIsDeviceKernel;
	const bool kernel_device_only = TConvolutionKernel::kIsDeviceKernel && !TConvolutionKernel::kIsHostKernel;

	static_assert(!(kernel_device_only && !views_device), "Cannot use device-only kernel on non-device views");
	static_assert(!(kernel_host_only && !views_host), "Cannot use host-only kernel on non-host views");

	if(TPolicy::kPreloadToSharedMemory){
		policy.setPreload(convolution_kernel.size(), convolution_kernel.center());
	}

	ConvolutionFunctor<typename KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::Type, typename TOutView::Element>
		functor{ KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::get(convolution_kernel) };

	detail::TransformLocatorPositionFunctor<decltype(functor), TInView, TOutView, TPolicy> lambda (functor, view_out);

	forEachLocator(
			view_in,
			lambda,
			policy,
			cuda_stream);
}

/// Convolves view with convolution kernel
/// Used mainly for testing ForeachLocator
/// \param view_in Input view, can be read only
/// \param view_out Output view - must provide write access
/// \param convolution_kernel Convolution kernel to be applied
/// \param cuda_stream Which stream should schedule this operation
template<typename TInView, typename TOutView, typename TConvolutionKernel>
void convolutionForeach(TInView view_in, TOutView view_out, const TConvolutionKernel & convolution_kernel, cudaStream_t cuda_stream = 0){
	convolutionForeach(view_in, view_out, convolution_kernel, DefaultTransformLocatorPolicy<TInView, TOutView>(), cuda_stream);
}

}  // namespace bolt
