// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

namespace bolt {

#ifdef __CUDA_ARCH__
template <typename TType>
static
inline
BOLT_DECL_HYBRID
auto
sqrt(const TType value) -> decltype(sqrtf(value))
{
	return sqrtf(value);
}
#else
template <typename TType>
static
inline
BOLT_DECL_HYBRID
auto
sqrt(const TType value) -> decltype(std::sqrt(value))
{
	return std::sqrt(value);
}
#endif

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator==(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	for (int i =  0; i < tDimension; ++i) {
		if (v1[i] != v2[i]) {
			return false;
		}
	}
	return true;
}

template<typename TType>
BOLT_DECL_HYBRID
bool operator==(const Vector<TType, 1> &v1, TType value){
	return v1[0] == value;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator!=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	return !(v1 == v2);
}

template<typename TType>
BOLT_DECL_HYBRID
bool operator!=(const Vector<TType, 1> &v1, TType value){
	return v1[0] != value;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator<(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	for (int i =  0; i < tDimension; ++i) {
		if (v1[i] >= v2[i]) {
			return false;
		}
	}
	return true;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator<(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2) {
	using common_type = Vector<typename std::common_type<TType1, TType2>::type, tDimension>;
	return static_cast<common_type>(v1) < static_cast<common_type>(v2);
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator>(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	return v2 < v1;
}


template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator<=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	for (int i =  0; i < tDimension; ++i) {
		if (v1[i] > v2[i]) {
			return false;
		}
	}
	return true;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator<=(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2) {
	using common_type = Vector<typename std::common_type<TType1, TType2>::type, tDimension>;
	return static_cast<common_type>(v1) <= static_cast<common_type>(v2);
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2) {
	return v2 <= v1;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2) {
	using common_type = Vector<typename std::common_type<TType1, TType2>::type, tDimension>;
	return static_cast<common_type>(v2) <= static_cast<common_type>(v1);
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator*(TType1 factor, const Vector<TType2, tDimension> &v) {
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	for (int i =  0; i < tDimension; ++i) {
		result[i] = factor * v[i];
	}

	return result;
}


template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2) {
	// TODO(johny) - expression templates
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	#ifdef __CUDA_ARCH__
		#pragma unroll
		for (int i =  0; i < tDimension; ++i) {
			result[i] = v1[i] + v2[i];
		}
	#else
		for (int i =  0; i < tDimension; ++i) {
			result[i] = v1[i] + v2[i];
		}
	#endif // __CUDA_ARCH__

	return result;
}


template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2) {
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	#ifdef __CUDA_ARCH__
		#pragma unroll
		for (int i =  0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#else
		for (int i =  0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#endif

	return result;
}


template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension>
operator-(const Vector<TType, tDimension> &v) {
	Vector<TType, tDimension> result;
	for (int i =  0; i < tDimension; ++i) {
		result[i] = -v[i];
	}
	return result;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension>
operator+(const Vector<TType, tDimension> &v) {
	return v;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const Vector<TType1, tDimension> &v1, const RepeatVector<TType2> &v2) {
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	#ifdef __CUDA_ARCH__
		#pragma unroll
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] + v2[i];
		}
	#else
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] + v2[i];
		}
	#endif // __CUDA_ARCH__

	return result;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const RepeatVector<TType1> &v1, const Vector<TType2, tDimension> &v2) {
	return v2 + v1;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const Vector<TType1, tDimension> &v1, const RepeatVector<TType2> &v2) {
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	#ifdef __CUDA_ARCH__
		#pragma unroll
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#else
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#endif // __CUDA_ARCH__

	return result;
}

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const RepeatVector<TType1> &v1, const Vector<TType2, tDimension> &v2) {
	Vector<typename std::common_type<TType1, TType2>::type, tDimension> result;
	#ifdef __CUDA_ARCH__
		#pragma unroll
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#else
		for (int i = 0; i < tDimension; ++i) {
			result[i] = v1[i] - v2[i];
		}
	#endif // __CUDA_ARCH__

	return result;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension> max(Vector<TType, tDimension> v1, const Vector<TType, tDimension> &v2) {
	for (int i =  0; i < tDimension; ++i) {
		v1[i] = v1[i] < v2[i] ? v2[i] : v1[i];
	}
	return v1;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension> min(Vector<TType, tDimension> v1, const Vector<TType, tDimension> &v2) {
	for (int i =  0; i < tDimension; ++i) {
		v1[i] = v1[i] < v2[i] ? v1[i] : v2[i];
	}
	return v1;
}


template<typename TVector1, typename TVector2>
BOLT_DECL_HYBRID
typename std::common_type<
	typename TVector1::Element,
	typename TVector2::Element>::type
dot(const TVector1 &v1, const TVector2 &v2) {
	static_assert(TVector1::kDimension == TVector2::kDimension,
		"The vectors must have the same dimension.");

	typename std::common_type<
		typename TVector1::Element,
		typename TVector2::Element
	>::type result = 0;
	for (int i = 0; i < TVector1::kDimension; ++i) {
		result += v1[i] * v2[i];
	}
	return result;
}

template<typename TType>
BOLT_DECL_HYBRID
inline TType dot(const Vector<TType, 3> &v1, const Vector<TType, 3> &v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> cross(const Vector<TType, 3> &v1, const Vector<TType, 3> &v2) {
	return Vector<TType, 3>(
		v1[1] * v2[2] - v1[2] * v2[1],
		v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]);
}

template<typename TElement1, typename TElement2, int tDimension>
BOLT_DECL_HYBRID
auto div(const Vector<TElement1, tDimension> &v1, const Vector<TElement2, tDimension> &v2) -> Vector<decltype(TElement1{} / TElement2{}), tDimension> {
	Vector<decltype(TElement1{} / TElement2{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = v1[i] / v2[i];
	}
	return result;
}

template<typename TElement, typename TScalar, int tDimension, class>
BOLT_DECL_HYBRID
auto div(const Vector<TElement, tDimension> &v1, const TScalar &scalar) -> Vector<decltype(TElement{} / TScalar{}), tDimension>{
	Vector<decltype(TElement{} / TScalar{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = v1[i] / scalar;
	}
	return result;
}

template<typename TElement, typename TScalar, int tDimension, class>
BOLT_DECL_HYBRID
auto div(const TScalar &scalar, const Vector<TElement, tDimension> &v1) -> Vector<decltype(TScalar{} / TElement{}), tDimension>{
	Vector<decltype(TScalar{} / TElement{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = scalar / v1[i];
	}
	return result;
}


template<typename TVector>
BOLT_DECL_HYBRID
TVector mod(const TVector &v1, const TVector &v2) {
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
		result[i] = v1[i] % v2[i];
	}
	return result;
}

template<typename TElement, typename TScalar, int tDimension, class>
BOLT_DECL_HYBRID
auto mod(const Vector<TElement, tDimension> &v1, const TScalar &scalar) -> Vector<decltype(TElement{} % TScalar{}), tDimension>{
	Vector<decltype(TElement{} % TScalar{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = v1[i] % scalar;
	}
	return result;
}

template<typename TElement, typename TScalar, int tDimension, class>
BOLT_DECL_HYBRID
auto mod(const TScalar &scalar, const Vector<TElement, tDimension> &v1) -> Vector<decltype(TScalar{} % TElement{}), tDimension>{
	Vector<decltype(TScalar{} % TElement{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = scalar % v1[i];
	}
	return result;
}

template<typename TVector>
BOLT_DECL_HYBRID
TVector modPeriodic(const TVector &v1, const TVector &v2) {
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
		result[i] = v1[i] % v2[i];
		result[i] += result[i] < 0 ? v2[i] : 0;
	}
	return result;
}

template<typename TElement1, typename TElement2, int tDimension>
BOLT_DECL_HYBRID
auto product(const Vector<TElement1, tDimension> &v1, const Vector<TElement2, tDimension> &v2) -> Vector<decltype(TElement1{} * TElement2{}), tDimension> {
	Vector<decltype(TElement1{} * TElement2{}), tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = v1[i] * v2[i];
	}
	return result;
}

template<typename TVector>
BOLT_DECL_HYBRID
typename TVector::Element product(const TVector &v) {
	typename TVector::Element result = 1;
	for (int i = 0; i < TVector::kDimension; ++i) {
		result *= v[i];
	}
	return result;
}

template<typename TVector, class, class>
BOLT_DECL_HYBRID
typename TVector::Element sum(const TVector &v) {
	typename TVector::Element result = 0;
	for (int i = 0; i < TVector::kDimension; ++i) {
		result += v[i];
	}
	return result;
}


/// Round all elements to neares integral values. Halves are rounded away from zero.
template<typename TVector, class, class>
BOLT_DECL_HYBRID
TVector round(const TVector &v){
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
#ifdef __CUDA_ARCH__
		result[i] = roundf(v[i]);
#else
		result[i] = std::round(v[i]);
#endif
	}
	return result;
}


template<typename TVector, class, class>
BOLT_DECL_HYBRID
TVector ceil(const TVector &v){
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
#ifdef __CUDA_ARCH__
		result[i] = ceilf(v[i]);
#else
		result[i] = std::ceil(v[i]);
#endif
	}
	return result;
}


template<typename TVector, class, class>
BOLT_DECL_HYBRID
TVector floor(const TVector &v){
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
#ifdef __CUDA_ARCH__
		result[i] = floorf(v[i]);
#else
		result[i] = std::floor(v[i]);
#endif
	}
	return result;
}

template<typename TVector, class, class>
BOLT_DECL_HYBRID
TVector abs(const TVector &v){
	TVector result;
	for (int i = 0; i < TVector::kDimension; ++i) {
#ifdef __CUDA_ARCH__
		result[i] = fabsf(v[i]);
#else
		result[i] = std::abs(v[i]);
#endif
	}
	return result;
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
int find(const Vector<TType, tDimension> &v, TType value) {
	for (int i = 0; i < tDimension; ++i) {
		if (v[i] == value) {
			return i;
		}
	}
	return -1;
}


template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension - 1>
removeDimension(const Vector<TType, tDimension> &v, int dimension) {
	BOLT_ASSERT(dimension >=0);
	BOLT_ASSERT(dimension < tDimension);
	Vector<TType, tDimension - 1> result;
	int result_index = 0;
	for (int i = 0; i < tDimension; ++i) {
		if (i != dimension) {
			result[result_index++] = v[i];
		}
	}
	return result;
}


template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension + 1>
insertDimension(const Vector<TType, tDimension> &v, TType inserted_coordinate, int dimension) {
	BOLT_ASSERT(dimension >=0);
	BOLT_ASSERT(dimension <= tDimension);
	Vector<TType, tDimension + 1> result;
	for (int i = 0; i < dimension; ++i) {
		result[i] = v[i];
	}
	result[dimension] = inserted_coordinate;
	for (int i = dimension; i < tDimension; ++i) {
		result[i + 1] = v[i];
	}
	return result;
}

template<typename TType, int tDimension>
std::ostream &operator<<(std::ostream &stream, const Vector<TType, tDimension> &v) {
	stream << "[" << v[0];
	for (int i = 1; i < tDimension; ++i) {
		stream << ", " << v[i];
	}
	return stream << "]";
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType minElement(const Vector<TType, tDimension> &val) {
	int min_index = 0;
	for (int i = 1; i < tDimension; ++i) {
		if (val[i] < val[min_index]) {
			min_index = i;
		}
	}
	return val[min_index];
}

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType maxElement(const Vector<TType, tDimension> &val) {
	int max_index = 0;
	for (int i = 1; i < tDimension; ++i) {
		if (val[i] > val[max_index]) {
			max_index = i;
		}
	}
	return val[max_index];
}


template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType minElementWithLimit(const Vector<TType, tDimension> &val, TType lower_limit, TType fallback) {
	int min_index = tDimension;
	for (int i = 0; i < tDimension; ++i) {
		if (val[i] > lower_limit) {
			min_index = i;
			break;
		}
	}
	for (int i = min_index + 1; i < tDimension; ++i) {
		if (val[i] < val[min_index] && val[i] > lower_limit) {
			min_index = i;
		}
	}
	if (min_index < tDimension) {
		return val[min_index];
	} else {
		return fallback;
	}

}

/// Maximum element of vector.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType maxElementWithLimit(const Vector<TType, tDimension> &val, TType upper_limit, TType fallback) {
	int max_index = tDimension;
	for (int i = 0; i < tDimension; ++i) {
		if (val[i] < upper_limit) {
			max_index = i;
			break;
		}
	}
	for (int i = max_index + 1; i < tDimension; ++i) {
		if (val[i] > val[max_index] && val[i] < upper_limit) {
			max_index = i;
		}
	}
	if (max_index < tDimension) {
		return val[max_index];
	} else {
		return fallback;
	}

}

// Clamps the vector inside specified lower and upper bound in each dimension
template<typename TVector, class>
BOLT_DECL_HYBRID
TVector clamp(const TVector& v, const TVector& lo, const TVector& hi)
{
	TVector out;
	for (int i = 0; i < v.kDimension; ++i) {
		out[i] = v[i] < lo[i] ? lo[i] : v[i] > hi[i] ? hi[i] : v[i];
	}
	return out;
}

template<int tDimension>
BOLT_DECL_HYBRID
Vector<float, tDimension>
normalize(const Vector<float, tDimension> &v) {
	using Vector_type = Vector<float, tDimension>;

	Vector_type result;
	typename Vector_type::Element norm_inv = 1.f / norm(v);
	result = norm_inv * v;
	return result;
}

template<typename TElement, int tDimension>
BOLT_DECL_HYBRID
auto
length(const Vector<TElement, tDimension> &v) -> decltype(sqrt(squaredNorm(v))) {
    return sqrt(squaredNorm(v));
}

template<typename TElement, int tDimension>
BOLT_DECL_HYBRID
auto
norm(const Vector<TElement, tDimension> &v) -> decltype(length(v)) {
    return length(v);
}


namespace detail {

template<int tIndex, int...tIndices>
struct FillVectorImpl {
	template<typename TInVector, typename TOutVector, typename... TValues>
	BOLT_DECL_HYBRID
	static void call(const TInVector &in, TOutVector &out, int current_index, TValues...values) {
		static_assert(tIndex < TInVector::kDimension, "Access index is out of valid index range for the input vector!");
		out[current_index] = in[tIndex];
		FillVectorImpl<tIndices...>::call(in, out, current_index + 1, values...);
	}
};

template<int...tIndices>
struct FillVectorImpl<kNew, tIndices...> {
	template<typename TInVector, typename TOutVector, typename TValue, typename... TValues>
	BOLT_DECL_HYBRID
	static void call(const TInVector &in, TOutVector &out, int current_index, TValue value, TValues...values) {
		out[current_index] = value;
		FillVectorImpl<tIndices...>::call(in, out, current_index + 1, values...);
	}
};

template<int tIndex>
struct FillVectorImpl<tIndex> {
	template<typename TInVector, typename TOutVector, typename... TValues>
	BOLT_DECL_HYBRID
	static void call(const TInVector &in, TOutVector &out, int current_index, TValues...) {
		out[current_index] = in[tIndex];
	}
};

template<>
struct FillVectorImpl<kNew> {
	template<typename TInVector, typename TOutVector, typename... TValues>
	BOLT_DECL_HYBRID
	static void call(const TInVector &in, TOutVector &out, int current_index, TValues...) {
		//out[current_index] = in[tIndex];
	}
};

}  // namespace detail

template<int...tIndices, typename... TValues, typename TVector>
BOLT_DECL_HYBRID
auto swizzle(const TVector &v, TValues... values) {
	static_assert(sizeof...(TValues) == 0, "TODO - dimension insertion not yet implemented!");
	Vector<typename TVector::Element, sizeof...(tIndices)> result;
	detail::FillVectorImpl<tIndices...>::call(v, result, 0, values...);
	return result;
}



}  // namespace bolt

namespace boost {
namespace serialization {

template<class TArchive, typename TType, int tDimension>
void serialize(TArchive &ar, bolt::Vector<TType, tDimension> &vector, const unsigned int  /*version*/) {
	ar & boost::serialization::make_array(vector.pointer(), tDimension);
}

} // namespace serialization
} // namespace boost
