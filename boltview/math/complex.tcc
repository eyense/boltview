// Copyright 2019 Eyen SE
// Authors: Adam Kubista adam.kubista@eyen.se, Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

namespace bolt {

/// \addtogroup FFT
/// @{

static constexpr float kMagnitudeEpsilon = 0.000001f;

template<typename TType>
inline TType makeComplex(float amplitude, const float phase = 0.0f);

// device

BOLT_DECL_HYBRID
inline
constexpr bool operator==(const cufftComplex & lhs, const cufftComplex & rhs) {
	return lhs.x == rhs.x && lhs.y == rhs.y;
}


BOLT_DECL_HYBRID
inline
cufftComplex operator*(float real, cufftComplex complex) {
	complex.x *= real;
	complex.y *= real;
	return complex;
}

BOLT_DECL_HYBRID
inline
cufftComplex operator*(cufftComplex complex, float real) {
	return operator*(real, complex);
}

BOLT_DECL_HYBRID
inline
cufftComplex operator/(cufftComplex complex, float real) {
	complex.x /= real;
	complex.y /= real;
	return complex;
}


BOLT_DECL_HYBRID
inline
cufftComplex operator*(const cufftComplex & lhs, const cufftComplex & rhs){
	cufftComplex result;
	result.x = (lhs.x * rhs.x) - (lhs.y * rhs.y);
	result.y = (lhs.x * rhs.y) + (lhs.y * rhs.x);
	return result;
}

BOLT_DECL_HYBRID
inline
cufftComplex operator+(const cufftComplex & lhs, const cufftComplex & rhs){
	cufftComplex result;
	result.x = lhs.x + rhs.x;
	result.y = lhs.y + rhs.y;
	return result;
}

BOLT_DECL_HYBRID
inline
cufftComplex operator-(const cufftComplex & value){
	return -1 * value;
}

BOLT_DECL_HYBRID
inline
cufftComplex operator-(const cufftComplex & lhs, const cufftComplex & rhs){
	return lhs + (-rhs);
}

BOLT_DECL_HYBRID
inline
cufftComplex operator/(const cufftComplex & lhs, const cufftComplex & rhs){
	cufftComplex result;
	float div = (rhs.x * rhs.x) + (rhs.y * rhs.y);
	result.x = ((lhs.x * rhs.x) + (lhs.y * rhs.y)) / div;
	result.y = ((lhs.y * rhs.x) - (lhs.x * rhs.y)) / div;
	return result;
}

/// Creates a complex number, negative amplitude shifts phase by PI naturally
template<>
BOLT_DECL_HYBRID
inline
cufftComplex makeComplex(float amplitude, const float phase){
	cufftComplex result;
	result.x = amplitude * cos(phase);
	result.y = amplitude * sin(phase);
	return result;
}

BOLT_DECL_HYBRID
inline
float magnitude(const cufftComplex & value){
	return sqrt((value.x * value.x) + (value.y * value.y));
}

BOLT_DECL_HYBRID
inline
float phase(const cufftComplex & value){
	// NOTE(fidli): @Robustness undefinded behavior, return 0? throw? crash?
	if(value.y == value.x && value.y == 0) { return 0;
}
	float r = atan2(value.y, value.x);
	if(r < 0){
		r += 2*kPi;
	}
	return r;
}

BOLT_DECL_HYBRID
inline
cufftComplex conjugate(const cufftComplex & value){
	cufftComplex result = value;
	result.y = -value.y;
	return result;
}

BOLT_DECL_HYBRID
inline
cufftComplex normalize(const cufftComplex & value){
	float length = magnitude(value);
	if(length < kMagnitudeEpsilon) {
		return {0,0};
	}

	return value / length;
}

BOLT_DECL_HYBRID
inline
cufftComplex hadamard(const cufftComplex & lhs, const cufftComplex & rhs){
	cufftComplex result;
	result.x = lhs.x*rhs.x;
	result.y = lhs.y*rhs.y;
	return result;
}

// host

BOLT_DECL_HYBRID
inline
HostComplexType operator*(float real, HostComplexType complex) {
	complex.x *= real;
	complex.y *= real;
	return complex;
}

BOLT_DECL_HYBRID
inline
	HostComplexType operator*(HostComplexType complex, float real) {
	return operator*(real, complex);
}

BOLT_DECL_HYBRID
inline
HostComplexType operator/(HostComplexType complex, float real) {
	complex.x /= real;
	complex.y /= real;
	return complex;
}

BOLT_DECL_HYBRID
inline
HostComplexType operator*(const HostComplexType & lhs, const HostComplexType & rhs){
	return {
		(lhs.x * rhs.x) - (lhs.y * rhs.y),
		(lhs.x * rhs.y) + (lhs.y * rhs.x)
	};
}

BOLT_DECL_HYBRID
inline
HostComplexType operator+(const HostComplexType & lhs, const HostComplexType & rhs){
	return {
		lhs.x + rhs.x,
		lhs.y + rhs.y
	};
}

BOLT_DECL_HYBRID
inline
HostComplexType operator-(const HostComplexType & value){
	return -1 * value;
}

BOLT_DECL_HYBRID
inline
HostComplexType operator-(const HostComplexType & lhs, const HostComplexType & rhs){
	return lhs + (-rhs);
}

BOLT_DECL_HYBRID
inline
HostComplexType operator/(const HostComplexType & lhs, const HostComplexType & rhs){
	float div = (rhs.x * rhs.x) + (rhs.y * rhs.y);
	return {
		((lhs.x * rhs.x) + (lhs.y * rhs.y)) / div,
		((lhs.y * rhs.x) - (lhs.x * rhs.y)) / div
	};
}

template<>
inline
HostComplexType makeComplex(float amplitude, const float phase){
	return {
		amplitude * cos(phase),
		amplitude * sin(phase)
	};
}

BOLT_DECL_HYBRID
inline
float magnitude(const HostComplexType & value){
	return sqrt((value.x * value.x) + (value.y * value.y));
}

BOLT_DECL_HYBRID
inline
float magSquared(const HostComplexType & value){
	return (value.x * value.x) + (value.y * value.y);
}

BOLT_DECL_HYBRID
inline
float magSquared(const DeviceComplexType & value){
	return (value.x * value.x) + (value.y * value.y);
}

BOLT_DECL_HYBRID
inline
float phase(const HostComplexType & value){
	// NOTE(fidli): @Robustness undefinded behavior, return 0? throw? crash?
	 if(value.y == value.x && value.y == 0) { return 0;
}
	float r = atan2(value.y, value.x);
	if(r < 0){
		r += 2*kPi;
	}
	return r;
}

BOLT_DECL_HYBRID
inline
HostComplexType conjugate(const HostComplexType & value){
	HostComplexType result = value;
	result.y = -value.y;
	return result;
}

BOLT_DECL_HYBRID
inline
HostComplexType normalize(const HostComplexType & value){
	float length = magnitude(value);
	if(length < kMagnitudeEpsilon) {
		return {0,0};
	}
	return value / length;
}

BOLT_DECL_HYBRID
inline
HostComplexType hadamard(const HostComplexType & lhs, const HostComplexType & rhs){
	return {
		lhs.x*rhs.x,
		lhs.y*rhs.y
	};
}

/// @}

}  // namespace bolt
