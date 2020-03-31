// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <type_traits>

#include <boltview/exceptions.h>

#include <boltview/math/vector.h>

namespace bolt {

/// \addtogroup Math
/// @{

template<typename TType>
class Quaternion: public Vector<TType, 4> {
public:
	BOLT_DECL_HYBRID
	Quaternion()
	{}

	BOLT_DECL_HYBRID
	Quaternion(TType a, TType b, TType c, TType d) :
		Vector<TType, 4>(a, b, c, d)
	{}

	BOLT_DECL_HYBRID
	Quaternion(TType a, const Vector<TType, 3> &v) :
		Vector<TType, 4>(a, v[0], v[1], v[2])
	{}

	BOLT_DECL_HYBRID
	Vector<TType, 3> getVector() const {
		return Vector<TType, 3>((*this)[1], (*this)[2], (*this)[3]);
	}
};


/// \addtogroup QuaternionOperations
/// @{
template<typename TType1, typename TType2>
BOLT_DECL_HYBRID
Quaternion<typename std::common_type<TType1, TType2>::type>
operator*(TType1 factor, const Quaternion<TType2> &q);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Quaternion<TType> &q1, const Quaternion<TType> &q2);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Vector<TType, 3> &v1, const Quaternion<TType> &q2);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Quaternion<TType> &q1, const Vector<TType, 3> &v2);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> conjugate(const Quaternion<TType> &q);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> inverted(const Quaternion<TType> &q);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> rotationQuaternion(TType angle, const Vector<TType, 3> &axis);

template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> rotation2VectorsToQuaternion(Vector<TType, 3> s, Vector<TType, 3> t);


template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> rotate(const Vector<TType, 3> &v, TType angle, const Vector<TType, 3> &axis);

template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> rotate(const Vector<TType, 3> &v, const Quaternion<TType>& rotation_quaterion);


inline Quaternion<float> eulerAnglesZXZToQuaternion(float z1, float x, float z2);

/// Hopf coordinates to quaternion. (theta, phi) are 2-sphere coordinates, psi is
/// the 1-sphere coordinate. Theta has range [0, PI], phi [0, 2PI), psi [0, 2PI)
template<typename TFloatType>
inline Quaternion<TFloatType> hopfCoordinatesToQuaternion(TFloatType theta, TFloatType phi, TFloatType psi);
template<typename TFloatType>
inline Quaternion<TFloatType> hopfCoordinatesToQuaternion(Vector<TFloatType, 3> hopf);
/// @}

/// @}

}  // namespace bolt

#include <boltview/math/quaternion.tcc>
