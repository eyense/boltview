// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE HybridQuaternionTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boltview/cuda_utils.h>
#include <boltview/math/quaternion.h>

#include <boltview/tests/test_utils.h>

namespace bolt {

template<typename TFloatType>
inline void quaternionCheckClose(Quaternion<TFloatType> a, Quaternion<TFloatType> b, TFloatType tolerance){
	for(int i = 0; i < a.kDimension; ++i){
		BOOST_CHECK_CLOSE(a[i], b[i], tolerance);
	}
}

BOLT_AUTO_TEST_CASE(VectorMultiplication) {
	Vector<float, 3> v(1.0f, 7.0f, 1.0f);
	Quaternion<float> q(2.0f, -3.0f, -4.65, 5.62);

	BOOST_CHECK_EQUAL((v * q), (Quaternion<float>(0.0f, v) * q));
	BOOST_CHECK_EQUAL((q * v), (q * Quaternion<float>(0.0f, v)));
}


BOLT_AUTO_TEST_CASE(Inversion) {
	Quaternion<float> q = rotationQuaternion(float(kPi / 2), Float3(0.0f, 1.0f , 0.0f));
	Quaternion<float> qi = inverted(q);
	Quaternion<float> qii = inverted(qi);

	float difference = squaredNorm((q * qi) - (qi * q));
	BOOST_CHECK_CLOSE(difference, 0.0f, kFloatTestEpsilon);

	difference = squaredNorm(q - qii);
	BOOST_CHECK_CLOSE(difference, 0.0f, kFloatTestEpsilon);
}


BOLT_AUTO_TEST_CASE(Rotation) {
	Vector<float, 3> v(1.0f, 0.0f, 0.0f);

	Vector<float, 3> result = rotate(v, float(kPi / 2.0f), Vector<float, 3>(0.0f, 1.0f, 0.0f));

	BOOST_CHECK_CLOSE(0.0f, result[0], kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(0.0f, result[1], kFloatTestEpsilon);
	BOOST_CHECK_CLOSE(-1.0f, result[2], 10 * kFloatTestEpsilon); // TODO(op): check this
}

BOLT_AUTO_TEST_CASE(HopfToQuaternion) {
	std::vector<Vector<float, 3>> hopf = {
		{1.57079632679489, 3.141592653589793,   1.57079632679489},
		{1.57079632679489, 0,                   1.57079632679489},
		{1.57079632679489, 3.141592653589793, 	4.71238898038469},
		{1.57079632679489, 0,                   4.71238898038469},
		{1.57079632679489, 3.141592653589793,   2.617993877991494},
		{1.57079632679489, 3.141592653589793,   5.759586531581287},
		{1.57079632679489, 0,                   0.5235987755982988},
		{1.57079632679489, 0,                   3.665191429188092}};
	std::vector<Quaternion<float>> quaKnown = {
		{ 0.5,                0.5,                -0.5,                -0.5},
		{ 0.5,                0.5,                 0.5,                 0.5},
		{-0.5,                0.5,                 0.5,                -0.5},
		{-0.5,                0.5,                -0.5,                 0.5},
		{0.1830127018922193,  0.6830127018922194, -0.1830127018922193, -0.6830127018922194},
		{-0.6830127018922194, 0.1830127018922193,  0.6830127018922194, -0.1830127018922193},
		{0.6830127018922194,  0.1830127018922193,  0.6830127018922194,  0.1830127018922193},
		{-0.1830127018922193, 0.6830127018922194, -0.1830127018922193,  0.6830127018922194}};

	for(int i = 0; i < hopf.size(); ++i){
		quaternionCheckClose(hopfCoordinatesToQuaternion(hopf[i]), quaKnown[i], kFloatTestEpsilonSinglePrecision);
	}
}

BOOST_AUTO_TEST_CASE(ZXZ){
	Vector<float, 3> v1(1, 0, 0);
	Vector<float, 3> result(-1, 0, 0);
	Quaternion<float> rotation = eulerAnglesZXZToQuaternion(1.57079632679489f, 0, 1.57079632679489f);
	Vector<float, 3> rotationResult = rotate(v1, rotation);
	testVectorsForIdentity(result, rotationResult);
}

BOOST_AUTO_TEST_CASE(ZYZ){
	Vector<float, 3> v1(1, 0, 0);
	Vector<float, 3> result(0, 0, -1);
	Quaternion<float> rotation = eulerAnglesZYZToQuaternion(1.57079632679489f, 1.57079632679489f, 0);
	Vector<float, 3> rotationResult = rotate(v1, rotation);
	testVectorsForIdentity(result, rotationResult);
}

}  // namespace bolt
