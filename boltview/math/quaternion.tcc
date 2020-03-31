// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

namespace bolt {


template<typename TType1, typename TType2>
BOLT_DECL_HYBRID
Quaternion<typename std::common_type<TType1, TType2>::type>
operator*(TType1 factor, const Quaternion<TType2> &q) {
	Quaternion<typename std::common_type<TType1, TType2>::type> result;
	for (int i =  0; i < 4; ++i) {
		result[i] = factor * q[i];
	}

	return result;
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Quaternion<TType> &q1, const Quaternion<TType> &q2) {
	Quaternion<TType> result;
	result[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3];
	result[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2];
	result[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1];
	result[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0];
	return result;
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Vector<TType, 3> &v1, const Quaternion<TType> &q2) {
	Quaternion<TType> result;
	result[0] = - v1[0] * q2[1] - v1[1] * q2[2] - v1[2] * q2[3];
	result[1] = + v1[0] * q2[0] + v1[1] * q2[3] - v1[2] * q2[2];
	result[2] = - v1[0] * q2[3] + v1[1] * q2[0] + v1[2] * q2[1];
	result[3] = + v1[0] * q2[2] - v1[1] * q2[1] + v1[2] * q2[0];
	return result;
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> operator*(const Quaternion<TType> &q1, const Vector<TType, 3> &v2) {
	Quaternion<TType> result;
	result[0] = - q1[1] * v2[0] - q1[2] * v2[1] - q1[3] * v2[2];
	result[1] = + q1[0] * v2[0] + q1[2] * v2[2] - q1[3] * v2[1];
	result[2] = + q1[0] * v2[1] - q1[1] * v2[2] + q1[3] * v2[0];
	result[3] = + q1[0] * v2[2] + q1[1] * v2[1] - q1[2] * v2[0];
	return result;
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> conjugate(const Quaternion<TType> &q) {
	return Quaternion<TType>(q[0], -q[1], -q[2], -q[3]);
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> inverted(const Quaternion<TType> &q) {
	Quaternion<TType> c = conjugate(q);
	return (1.0f / dot(q, q)) * c;
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> rotationQuaternion(TType angle, const Vector<TType, 3> &axis) {
	return Quaternion<TType>(static_cast<TType>(cos(angle / 2)), static_cast<TType>(sin(angle / 2)) * axis);
}


template<typename TType>
BOLT_DECL_HYBRID
Quaternion<TType> rotation2VectorsToQuaternion(Vector<TType, 3> s, Vector<TType, 3> t) {
	// use trigonometrical identities between angle and halfangle sines/cosines
	s = normalize(s);
	t = normalize(t);
	auto dp = dot(s, t);
	Quaternion<TType> q(
			sqrt((1.0f + dp) * 0.5f),
			(1.0f / sqrt(2.0f * (1.0f + dp))) * cross(s, t));
	return q;
}

template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> rotate(const Vector<TType, 3> &v, TType angle, const Vector<TType, 3> &axis) {
	Quaternion<TType> q = rotationQuaternion(angle, axis);
	return rotate(v, q);
}

template<typename TType>
BOLT_DECL_HYBRID
Vector<TType, 3> rotate(const Vector<TType, 3> &v, const Quaternion<TType>& rotation_quaternion){
	Quaternion<TType> rotation_quaternion_conjugate = conjugate(rotation_quaternion);
	Quaternion<TType> vq(0, v);
	Quaternion<TType> result = rotation_quaternion * vq * rotation_quaternion_conjugate;

	return Vector<TType, 3>(result[1], result[2], result[3]);
}


inline Quaternion<float> eulerAnglesZXZToQuaternion(float z1, float x, float z2) {
	return rotationQuaternion(z1, Float3(0.0f, 0.0f, 1.0f))
		* rotationQuaternion(x, Float3(1.0f, 0.0f, 0.0f))
		* rotationQuaternion(z2, Float3(0.0f, 0.0f, 1.0f));
}

inline Quaternion<float> eulerAnglesZYZToQuaternion(float z1, float y, float z2) {
        return rotationQuaternion(z1, Float3(0.0f, 0.0f, 1.0f))
                * rotationQuaternion(y, Float3(0.0f, 1.0f, 0.0f))
                * rotationQuaternion(z2, Float3(0.0f, 0.0f, 1.0f));
}

template<typename TFloatType>
inline Quaternion<TFloatType> hopfCoordinatesToQuaternion(TFloatType theta, TFloatType phi, TFloatType psi) {
	return Quaternion<TFloatType>(cos(0.5 * theta) * cos(0.5 * psi),
							 	  cos(0.5 * theta) * sin(0.5 * psi),
							 	  sin(0.5 * theta) * cos(phi + 0.5 * psi),
								  sin(0.5 * theta) * sin(phi + 0.5 * psi));
}

template<typename TFloatType>
inline Quaternion<TFloatType> hopfCoordinatesToQuaternion(Vector<TFloatType, 3> hopf){
	return hopfCoordinatesToQuaternion<TFloatType>(hopf[0], hopf[1], hopf[2]);
}


}  // namespace bolt
