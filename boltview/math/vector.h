// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#ifdef __CUDACC__
#include <cuda.h>
#endif  // __CUDACC__

#include <cmath>
#include <type_traits>

#include <boltview/cuda_defines.h>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/array.hpp>

namespace bolt {

/// \addtogroup Math
/// @{

constexpr double kPi = 3.141592653589793238462;

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
TT abs(const TT &v){
#ifdef __CUDA_ARCH__
		return fabsf(v);
#else
		return std::abs(v);
#endif
}

BOLT_DECL_HYBRID
inline float ceil(float f) {
#ifdef __CUDA_ARCH__
	return ceilf(f);
#else
	return std::ceil(f);
#endif
}

BOLT_DECL_HYBRID
inline float floor(float f) {
#ifdef __CUDA_ARCH__
	return floorf(f);
#else
	return std::floor(f);
#endif
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
TT round(const TT &v){
#ifdef __CUDA_ARCH__
		return roundf(v);
#else
		return std::round(v);
#endif
}


template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
constexpr TT power(TT v, uint32_t exp){
	TT res = 1;
	for (int i = 0; i < exp; ++i) {
		res *= v;
	}
	return res;
}



template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT square(TT val) {
	return val * val;
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT sqr(TT val) {
	return square(val);
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT cube(TT val) {
	return val * val * val;
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT min(TT a, TT b) {
	return a < b ? a : b;
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT max(TT a, TT b) {
	return a < b ? b : a;
}

template<typename TT, class = typename std::enable_if<std::is_scalar<TT>::value>::type>
BOLT_DECL_HYBRID
inline TT clamp(TT a, TT lo, TT hi) {
	if (a <= lo) {
		return lo;
	}
	if (a >= hi) {
		return hi;
	}
	return a;
}


namespace detail {

// Prevents warnings for unsigned types
template <typename TT>
BOLT_DECL_HYBRID
inline constexpr int signum(TT x, std::false_type  /*is_signed*/) {
	return TT(0) < x;
}

template <typename TT>
BOLT_DECL_HYBRID
inline constexpr int signum(TT x, std::true_type  /*is_signed*/) {
	return (TT(0) < x) - (x < TT(0));
}

}  // namespace detail

/// \return (-1, 0, 1) depending on the sign of the passed number
template <typename TT>
BOLT_DECL_HYBRID
inline constexpr int signum(TT x) {
	return detail::signum(x, std::is_signed<TT>());
}

struct FillTag{};

/// Staticaly sized arithmetic vector usable in host/device code.
/// TODO(johny) if we find some suitable hybrid vector implementation use it instead of this one
template<typename TType, int tDimension>
class Vector {
public:
	static const int kDimension = tDimension;
	using Element = TType;
        using value_type = Element;

	BOLT_DECL_HYBRID
	Vector() {
		for (int i = 0; i < tDimension; ++i) {
			data[i] = 0;
		}
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	explicit Vector(const TOtherType *buffer) {
		for (int i = 0; i < tDimension; ++i) {
			// static_cast prevents warnings
			// since assignment to different vector type is explicit operation in user responsibility
			data[i] = static_cast<TType>(buffer[i]);
		}
	}

	BOLT_DECL_HYBRID
	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions in case we have scalar value
	/*explicit*/ Vector(TType x) {
		static_assert(tDimension == 1, "Vector must have dimensionality 1");
		data[0] = x;
		// For enabling default trailing dimensions uncomment this and change the static_assert above
		/*for (int i = 1; i < tDimension; ++i) {
			data[i] = 0;
		}*/
	}

	BOLT_DECL_HYBRID
	Vector(TType v, FillTag /*tag*/) {
		for (int i = 0; i < tDimension; ++i) {
			data[i] = v;
		}
	}


	BOLT_DECL_HYBRID
	Vector(TType x, TType y) {
		static_assert(tDimension == 2, "Vector must have dimensionality 2");
		data[0] = x;
		data[1] = y;
		// For enabling default trailing dimensions uncomment this and change the static_assert above
		/*for (int i = 2; i < tDimension; ++i) {
			data[i] = 0;
		}*/
	}

	BOLT_DECL_HYBRID
	Vector(TType x, TType y, TType z) {
		static_assert(tDimension == 3, "Vector must have dimensionality 3");
		data[0] = x;
		data[1] = y;
		data[2] = z;
		// For enabling default trailing dimensions uncomment this and change the static_assert above
		/*for (int i = 3; i < tDimension; ++i) {
			data[i] = 0;
		}*/
	}

	BOLT_DECL_HYBRID
	Vector(TType x, TType y, TType z, TType w) {
		static_assert(tDimension == 4, "Vector must have dimensionality 4");
		data[0] = x;
		data[1] = y;
		data[2] = z;
		data[3] = w;
		// For enabling default trailing dimensions uncomment this and change the static_assert above
		/*for (int i = 4; i < tDimension; ++i) {
			data[i] = 0;
		}*/
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	explicit Vector(const Vector<TOtherType, tDimension> &v) {
		for (int i = 0; i < tDimension; ++i) {
			data[i] = static_cast<TType>(v[i]);
		}
	}

	template<int tOtherDim>
	BOLT_DECL_HYBRID
	explicit Vector(const Vector<TType, tOtherDim> &v, TType a) {
		static_assert(tDimension == tOtherDim + 1, "Incorrect vector dimension");
		for (int i = 0; i < tOtherDim; ++i) {
			data[i] = v[i];
		}
		data[tOtherDim] = a;
	}

	template<int tOtherDim>
	BOLT_DECL_HYBRID
	explicit Vector(const Vector<TType, tOtherDim> &v, TType a, TType b) {
		static_assert(tDimension == tOtherDim + 2, "Incorrect vector dimension");
		for (int i = 0; i < tOtherDim; ++i) {
			data[i] = v[i];
		}
		data[tOtherDim] = a;
		data[tOtherDim + 1] = b;
	}

	/// Lost precision when assigning different vector type is user responsibility
	template<typename TOtherType>
	BOLT_DECL_HYBRID
	Vector<TType, tDimension> &operator=(const Vector<TOtherType, tDimension> &v) {
		for (int i = 0; i < tDimension; ++i) {
			// static_cast prevents warnings
			// since assignment to different vector type is explicit operation in user responsibility
			data[i] = static_cast<TType>(v[i]);
		}
		return *this;
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	Vector<TType, tDimension> &operator+=(const Vector<TOtherType, tDimension> &v) {
		#ifdef __CUDA_ARCH__
			#pragma unroll
			for (int i =  0; i < tDimension; ++i) {
				data[i] += v[i];
			}
		#else
			for (int i =  0; i < tDimension; ++i) {
				data[i] += v[i];
			}
		#endif // __CUDA_ARCH__
		return *this;
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	Vector<TType, tDimension> &operator-=(const Vector<TOtherType, tDimension> &v) {
		#ifdef __CUDA_ARCH__
			#pragma unroll
			for (int i =  0; i < tDimension; ++i) {
				data[i] -= v[i];
			}
		#else
			for (int i =  0; i < tDimension; ++i) {
				data[i] -= v[i];
			}
		#endif // __CUDA_ARCH__
		return *this;
	}

	BOLT_DECL_HYBRID
	Vector<TType, tDimension> &operator/=(const TType &divisor) {
		#ifdef __CUDA_ARCH__
			#pragma unroll
			for (int i =  0; i < tDimension; ++i) {
				data[i] /= divisor;
			}
		#else
			for (int i =  0; i < tDimension; ++i) {
				data[i] /= divisor;
			}
		#endif // __CUDA_ARCH__
		return *this;
	}


	/// \return Unchecked element access
	BOLT_DECL_HYBRID
	TType &operator[](int index) {
		return data[index];
	}

	/// \return Unchecked element access
	BOLT_DECL_HYBRID
	TType operator[](int index) const {
		return data[index];
	}

	BOLT_DECL_HYBRID
	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions, but only for scalar types
	operator TType() const {
		static_assert(tDimension == 1, "Casting to scalar possible only for vectors of dimensionality 1");
		return data[0];
	}

	constexpr int dimension() const {
		return kDimension;
	}

	/// \return Buffer pointer (no ownership transfer)
	BOLT_DECL_HYBRID
	TType *pointer() {
		return data;
	}

	/// \return Buffer pointer (no ownership transfer)
	BOLT_DECL_HYBRID
	const TType *pointer() const {
		return data;
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	static Vector<TType, tDimension> fill(TOtherType value) {
		Vector<TType, tDimension> result;
		for (int i = 0; i < tDimension; ++i) {
			result[i] = value;
		}
		return result;
	}

protected:
	TType data[kDimension];
};

template <typename TVector>
struct VectorTraits
{
	static constexpr bool kIsVector = false;
};

template <typename TType, int tDim>
struct VectorTraits<Vector<TType, tDim>>
{
	static constexpr bool kIsVector = true;
	static constexpr int kDimension = tDim;

	using Element = TType;
	using This = Vector<TType, tDim>;
	using type = typename std::conditional<tDim == 1, Element, This>::type;
};

template<typename TVector>
struct IsVector : std::integral_constant<bool, VectorTraits<TVector>::kIsVector> {};


/// Vector with the same value in all elements usable in host/device code.
template<typename TType>
class RepeatVector {
public:
	using Element = TType;

	BOLT_DECL_HYBRID
	RepeatVector() : value_{0} {}

	BOLT_DECL_HYBRID
	explicit RepeatVector(TType scalar) {
		value_ = scalar;
	}

	template<typename TOtherType>
	BOLT_DECL_HYBRID
	explicit RepeatVector(const RepeatVector<TOtherType> &v) {
		value_ = v.value_;
	}

	/// \return The value.
	BOLT_DECL_HYBRID
	TType operator[](int /*index*/) const {
		return value_;
	}

protected:
	TType value_;
};

template <typename TType>
struct VectorTraits<RepeatVector<TType>>
{
	static constexpr bool kIsVector = true;
	static constexpr int kDimension = -1;

	using Element = TType;

};

/// Creates the RepeateVector object.
template<typename TType>
RepeatVector<TType> rep(TType scalar) {
	return RepeatVector<TType>(scalar);
}

/// \addtogroup VectorOperations
/// @{
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator==(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType>
BOLT_DECL_HYBRID
bool operator==(const Vector<TType, 1> &v1, TType value);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator!=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType>
BOLT_DECL_HYBRID
bool operator!=(const Vector<TType, 1> &v1, TType value);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator<(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator<(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator>(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator<=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator<=(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const Vector<TType, tDimension> &v1, const Vector<TType, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator*(TType1 factor, const Vector<TType2, tDimension> &v);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const Vector<TType1, tDimension> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension>
operator-(const Vector<TType, tDimension> &v);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension>
operator+(const Vector<TType, tDimension> &v);

// TODO(jakub): Consider more general way of operator+/- which enables to unify it with operator+/- for hybrid vectors (some sort of concepts).
template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const Vector<TType1, tDimension> &v1, const RepeatVector<TType2> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator+(const RepeatVector<TType1> &v1, const Vector<TType2, tDimension> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const Vector<TType1, tDimension> &v1, const RepeatVector<TType2> &v2);

template<typename TType1, typename TType2, int tDimension>
BOLT_DECL_HYBRID
Vector<typename std::common_type<TType1, TType2>::type, tDimension>
operator-(const RepeatVector<TType1> &v1, const Vector<TType2, tDimension> &v2);

/// Generic dot product
template<typename TVector1, typename TVector2>
BOLT_DECL_HYBRID
typename std::common_type<
	typename TVector1::Element,
	typename TVector2::Element>::type
dot(const TVector1 &v1, const TVector2 &v2);

/// Size of the vector squared
template<typename TVector>
BOLT_DECL_HYBRID
typename TVector::Element squaredNorm(const TVector &v) {
	return dot(v, v);
}

/// alias for length
template<typename TVector>
BOLT_DECL_HYBRID
float norm(const TVector &v);

template<typename TVector>
BOLT_DECL_HYBRID
float length(const TVector &v);

/// Generic cross product
template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> cross(const Vector<TType, 3> &v1, const Vector<TType, 3> &v2);

/// \return Per element maximum
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension> max(Vector<TType, tDimension> v1, const Vector<TType, tDimension> &v2);

/// \return Per element minimum
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension> min(Vector<TType, tDimension> v1, const Vector<TType, tDimension> &v2);

/// \return Per element division - rules same as for number division.
template<typename TElement1, typename TElement2, int tDimension>
BOLT_DECL_HYBRID
auto div(const Vector<TElement1, tDimension> &v1, const Vector<TElement2, tDimension> &v2) -> Vector<decltype(TElement1{} / TElement2{}), tDimension>;

template<typename TElement, typename TScalar, int tDimension, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
auto div(const Vector<TElement, tDimension> &v1, const TScalar &scalar) -> Vector<decltype(TElement{} / TScalar{}), tDimension>;

template<typename TElement, typename TScalar, int tDimension, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
auto div(const TScalar &scalar, const Vector<TElement, tDimension> &v1) -> Vector<decltype(TScalar{} / TElement{}), tDimension>;

/// \return Per element modulo
template<typename TVector>
BOLT_DECL_HYBRID
TVector mod(const TVector &v1, const TVector &v2);

template<typename TElement, typename TScalar, int tDimension, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
auto mod(const Vector<TElement, tDimension> &v1, const TScalar &scalar) -> Vector<decltype(TElement{} % TScalar{}), tDimension>;

template<typename TElement, typename TScalar, int tDimension, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
auto mod(const TScalar &scalar, const Vector<TElement, tDimension> &v1) -> Vector<decltype(TScalar{} % TElement{}), tDimension>;

template<typename TVector>
BOLT_DECL_HYBRID
TVector modPeriodic(const TVector &v1, const TVector &v2);


/// \return Per element product
template<typename TElement1, typename TElement2, int tDimension>
BOLT_DECL_HYBRID
auto product(const Vector<TElement1, tDimension> &v1, const Vector<TElement2, tDimension> &v2) -> Vector<decltype(TElement1{} * TElement2{}), tDimension>;


/// \return All vector elements product
template<typename TVector>
BOLT_DECL_HYBRID
typename TVector::Element product(const TVector &v);

template<typename TScalar, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
TScalar product(const TScalar &v) { return v; }


/// \return Sum of all vector elements
template<typename TVector, class = typename std::enable_if<IsVector<TVector>::value>::type, class = void>
BOLT_DECL_HYBRID
typename TVector::Element sum(const TVector &v);

inline BOLT_DECL_HYBRID
int64_t sum(const Vector<int64_t, 2> &v) {
	return v[0] + v[1];
}

inline BOLT_DECL_HYBRID
int64_t sum(const Vector<int64_t, 3> &v) {
	return v[0] + v[1] + v[2];
}

/// Round all elements to nearest integral values. Halves are rounded away from zero.
template<typename TVector, class = typename std::enable_if<IsVector<TVector>::value>::type, class = void>
BOLT_DECL_HYBRID
TVector round(const TVector &v);

/// Round up all elements to nearest bigger integral values.
template<typename TVector, class = typename std::enable_if<IsVector<TVector>::value>::type, class = void>
BOLT_DECL_HYBRID
TVector ceil(const TVector &v);

/// Round up all elements to nearest lower integral values.
template<typename TVector, class = typename std::enable_if<IsVector<TVector>::value>::type, class = void>
BOLT_DECL_HYBRID
TVector floor(const TVector &v);

/// Absolute value for all elements
template<typename TVector, class = typename std::enable_if<IsVector<TVector>::value>::type, class = void>
BOLT_DECL_HYBRID
TVector abs(const TVector &v);

/// \return Index of first element equal to passed value, or -1 when none is present.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
int find(const Vector<TType, tDimension> &v, TType value);

/// Creates new vector (of dimension smaller by 1) with same values except the selected one -> it removes one coordinate.
/// In other words it is a vector projection.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension - 1>
removeDimension(const Vector<TType, tDimension> &v, int dimension);

/// Insert new coordinate in the vector, thus increasing its dimension by 1.
/// \param v Original vector
/// \param value Inserted value
/// \param dimesion Before which coordinate we insert the new one
/// \result Vector of increased dimesnionality
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
Vector<TType, tDimension + 1>
insertDimension(const Vector<TType, tDimension> &v, TType inserted_coordinate, int dimension);

/// get type of Vector with one less dimensions in ::type
template<typename TVector>
struct DecreaseDimension {
	static const int kDimension = TVector::kDimension;
	using type = Vector<typename TVector::Element, kDimension - 1>;
};
template<typename TVector>
BOLT_DECL_HYBRID
typename TVector::Element get(const TVector &v, int index) { return v[index]; }

template<typename TVector>
BOLT_DECL_HYBRID
void set(TVector &v, int index, typename TVector::Element value) { v[index] = value; }

template<typename TScalar, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
TScalar get(const TScalar &v, int  /*index*/) { return v; }

template<typename TScalar, class = typename std::enable_if<std::is_scalar<TScalar>::value>::type>
BOLT_DECL_HYBRID
void set(TScalar &v, int  /*index*/, TScalar value) { v = value; }


template<typename TType, int tDimension>
std::ostream &operator<<(std::ostream &stream, const Vector<TType, tDimension> &v);

/// Normalize float vector
template<int tDimension>
BOLT_DECL_HYBRID
Vector<float, tDimension> normalize(const Vector<float, tDimension> &v);

/// Minimum element of vector.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType minElement(const Vector<TType, tDimension> &val);

/// Maximum element of vector.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType maxElement(const Vector<TType, tDimension> &val);

template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType minElementWithLimit(const Vector<TType, tDimension> &val, TType lower_limit, TType fallback);

/// Maximum element of vector.
template<typename TType, int tDimension>
BOLT_DECL_HYBRID
TType maxElementWithLimit(const Vector<TType, tDimension> &val, TType upper_limit, TType fallback);

// Clamps the vector inside specified lower and upper bound in each dimension
template<typename TVector, class = typename std::enable_if<!std::is_scalar<TVector>::value>::type>
BOLT_DECL_HYBRID
TVector clamp(const TVector& v, const TVector& lo, const TVector& hi);

/// @}

/// \addtogroup VectorTypedefs
/// @{
using Int1 = Vector<int32_t, 1>;
using Int2 = Vector<int32_t, 2>;
using Int3 = Vector<int32_t, 3>;
using Int4 = Vector<int32_t, 4>;
using Bool1 = Vector<bool, 1>;
using Bool2 = Vector<bool, 2>;
using Bool3 = Vector<bool, 3>;
using Bool4 = Vector<bool, 4>;
using Float1 = Vector<float, 1>;
using Float2 = Vector<float, 2>;
using Float3 = Vector<float, 3>;
using Float4 = Vector<float, 4>;
using LongInt1 = Vector<int64_t, 1>;
using LongInt2 = Vector<int64_t, 2>;
using LongInt3 = Vector<int64_t, 3>;
using LongInt4 = Vector<int64_t, 4>;

/// @}
/// @}

/// \addtogroup Utilities
/// @{

/// \addtogroup ExceptionErrorInfo
/// @{
using Index3DErrorInfo = boost::error_info<struct tag_index_3d, Int3>;
using Index2DErrorInfo = boost::error_info<struct tag_index_2d, Int2>;
/// @}
/// @}

template<int tDimension, typename TElement = int>
BOLT_DECL_HYBRID
inline int64_t linearIndex(
	const Vector<TElement, tDimension> strides,
	const Vector<TElement, tDimension> index)
{
	#ifdef __CUDA_ARCH__
		int64_t linear_index = 0;
		#pragma unroll
		for (int i =  0; i < tDimension; ++i) {
			linear_index += int64_t(strides[i]) * index[i];
		}
		return linear_index;
	#else
		int64_t linear_index = 0;
		for (int i =  0; i < tDimension; ++i) {
			linear_index += int64_t(strides[i]) * index[i];
		}
		return linear_index;
	#endif // __CUDA_ARCH__
}

/// @}


template <typename TType, int tDimension>
BOLT_DECL_HYBRID
inline TType *begin(Vector<TType, tDimension> &vector) {
	return vector.pointer();
}

template <typename TType, int tDimension>
BOLT_DECL_HYBRID
inline TType *end(Vector<TType, tDimension> &vector) {
	return vector.pointer() + tDimension;
}

template <typename TType, int tDimension>
BOLT_DECL_HYBRID
inline const TType *begin(const Vector<TType, tDimension> &vector) {
	return vector.pointer();
}

template <typename TType, int tDimension>
BOLT_DECL_HYBRID
inline const TType *end(const Vector<TType, tDimension> &vector) {
	return vector.pointer() + tDimension;
}


/// Special value to indicate where new value is inserted into the vector in the swizzle function
static constexpr int kNew = -1;

template<int...tIndices, typename... TValues, typename TVector>
BOLT_DECL_HYBRID
auto swizzle(const TVector &v, TValues... values);

}  // namespace bolt

#include <boltview/math/vector.tcc>
