// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se

#pragma once

#ifdef BOLT_ENABLE_FFT
#include <complex>
#include <cufft.h>
#include <fftw3.h>
#endif

namespace bolt {

#ifdef BOLT_ENABLE_FFT
using DeviceComplexType = cufftComplex;
union HostComplexType{
	fftwf_complex fftw;
	cufftComplex cuda;
	struct{
		float x;
		float y;
	};

	BOLT_DECL_HYBRID
	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions
	operator cufftComplex() const {
		return cuda;
	}
};
#else
struct Complex{
    float x;
    float y;
};
using DeviceComplexType = Complex;
using HostComplexType = Complex;
#endif

}  // namespace bolt

#include <boltview/math/complex.tcc>
