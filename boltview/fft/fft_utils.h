// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se

#pragma once


#include <boltview/math/complex.h>
#include <boltview/math/vector.h>
#include <boltview/cuda_defines.h>
#include <boltview/fft/fft_calculator.h>

#if defined(__CUDACC__)
#include <boltview/cuda_utils.h>
#include <boltview/device_image.h>
#include <boltview/device_image_view.h>
#endif  // defined(__CUDACC__)

#include <boltview/host_image.h>
#include <boltview/procedural_views.h>
#include <boltview/transform.h>

#include <boltview/array_view.h>
#include <boltview/view_traits.h>


namespace bolt {

/// \addtogroup FFT
/// @{

/// Calculates the cross power spectrum of two views
template<typename TInputViewA, typename TInputViewB>
auto crossPowerSpectrum(TInputViewA view_a, TInputViewB view_b){
	static_assert((TInputViewA::kIsHostView && TInputViewB::kIsHostView) || (TInputViewA::kIsDeviceView && TInputViewB::kIsDeviceView), "Both views must be in the same memory");
	return normalize(multiply(view_a, conjugate(view_b)));
}

#ifdef __CUDACC__

/// Executes Phase correlation form space domain images into space domain, runs on device
/// TODO(johny) stack policy
template<typename TInputViewA, typename TInputViewB, typename ::std::enable_if<IsDeviceView<TInputViewA>::value && IsDeviceView<TInputViewB>::value && TInputViewA::kIsMemoryBased && TInputViewB::kIsMemoryBased>::type * = nullptr>
DeviceImage<float, TInputViewA::kDimension> phaseCorrelation(TInputViewA viewA, TInputViewB viewB){
	if(viewA.size() != viewB.size()){
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(viewA.size(), viewB.size()));
	}
	FftCalculator<TInputViewA::kDimension, DeviceFftPolicy<Forward>> forward(viewA.size());

	auto frequencyA = forward.createFrequencyDomainDeviceImage();
	auto frequencyB = forward.createFrequencyDomainDeviceImage();

	forward.calculateAndNormalize(viewA, frequencyA.view());
	forward.calculateAndNormalize(viewB, frequencyB.view());

	auto cross_power_spectrum  = crossPowerSpectrum(frequencyA.view(), frequencyB.view());

	FftCalculator<TInputViewA::kDimension, DeviceFftPolicy<Inverse>> inverse(viewA.size());
	auto cross_power = inverse.createFrequencyDomainDeviceImage();
	copy(cross_power_spectrum, cross_power.view());
	auto correlation = inverse.createSpaceDomainDeviceImage();
	inverse.calculate(cross_power.view(), correlation.view());
	return correlation;
}


#endif  // __CUDACC__

/// Executes phase correlation form space domain images into space domain, runs on host
/// TODO(johny) stack policy
template<typename TInputViewA, typename TInputViewB, typename ::std::enable_if<IsHostView<TInputViewA>::value && IsHostView<TInputViewB>::value>::type * = nullptr>
HostImage<float, TInputViewA::kDimension> phaseCorrelation(TInputViewA view_a, TInputViewB view_b){
	if(view_a.size() != view_b.size()){
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(view_a.size(), view_b.size()));
	}
	FftCalculator<TInputViewA::kDimension, HostFftPolicy<Forward>> forward(view_a.size());
	auto frequency_a = forward.createFrequencyDomainHostImage();
	auto frequency_b = forward.createFrequencyDomainHostImage();

	forward.calculateAndNormalize(view_a, frequency_a.view());
	forward.calculateAndNormalize(view_b, frequency_b.view());

	auto cross_power_spectrum = crossPowerSpectrum(frequency_a.view(), frequency_b.view());

	FftCalculator<TInputViewA::kDimension, HostFftPolicy<Inverse>> inverse(view_a.size());
	auto correlation = inverse.createSpaceDomainHostImage();
	inverse.calculate(cross_power_spectrum, correlation.view());
	return correlation;
}


/// @}

}  // namespace bolt

#include <boltview/fft/fft_utils.tcc>
