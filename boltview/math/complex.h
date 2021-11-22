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
using DeviceDoubleComplexType = cufftDoubleComplex;
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
struct DeviceDoubleComplexType{
    double x;
    double y;
};
union HostComplexType{
	Complex fftw;
	Complex cuda;
	struct{
		float x;
		float y;
	};
};

#endif

}  // namespace bolt

#include <boltview/math/complex.tcc>
