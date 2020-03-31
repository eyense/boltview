// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se

#pragma once

#include <complex>
#include <cufft.h>
#include <fftw3.h>

namespace bolt {

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

}  // namespace bolt

#include <boltview/math/complex.tcc>
