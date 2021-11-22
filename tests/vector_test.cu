// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE HybridVectorTest
#include <boost/test/included/unit_test.hpp>
#include <boltview/tests/test_utils.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include <boltview/cuda_utils.h>
#include <boltview/math/quaternion.h>


namespace bolt {

BOLT_AUTO_TEST_CASE(HostInitialization) {
	Int3 v1(1, 2, 3);
	BOOST_CHECK_EQUAL(v1[0], 1);
	BOOST_CHECK_EQUAL(v1[1], 2);
	BOOST_CHECK_EQUAL(v1[2], 3);

	Float3 v2;
	BOOST_CHECK_EQUAL(v2[0], 0);
	BOOST_CHECK_EQUAL(v2[1], 0);
	BOOST_CHECK_EQUAL(v2[2], 0);
	v2 = v1;
	BOOST_CHECK_EQUAL(v2[0], 1);
	BOOST_CHECK_EQUAL(v2[1], 2);
	BOOST_CHECK_EQUAL(v2[2], 3);

	Float3 v3(v1);
	BOOST_CHECK_EQUAL(v3[0], 1);
	BOOST_CHECK_EQUAL(v3[1], 2);
	BOOST_CHECK_EQUAL(v3[2], 3);
}

BOLT_AUTO_TEST_CASE(Comparison) {
	Int3 v1(1, 2, 3);
	Int3 v1_clone(v1);
	Int3 v2(3, 5, -1);
	BOOST_CHECK_EQUAL(v1, v1);
	BOOST_CHECK(!(v1 != v1));
	BOOST_CHECK_EQUAL(v1, v1_clone);
	BOOST_CHECK(!(v1 != v1_clone));

	BOOST_CHECK(v1 != v2);
	BOOST_CHECK(!(v1 == v2));

	BOOST_CHECK(v2 != v1);
	BOOST_CHECK(!(v2 == v1));
}


BOLT_AUTO_TEST_CASE(Ordering) {
	Int3 v1(1, 2, 3);
	Int3 v2(2, 3, 4);
	Int3 v3(2, 3, 3);
	Int3 v1_clone = v1;

	BOOST_CHECK(v1 <= v1);
	BOOST_CHECK(v1 >= v1);
	BOOST_CHECK(v1 <= v1_clone);
	BOOST_CHECK(v1 >= v1_clone);
	BOOST_CHECK(v1 < v2);
	BOOST_CHECK(v2 > v1);
	BOOST_CHECK(!(v1 > v2));
	BOOST_CHECK(!(v1 < v3));
	BOOST_CHECK(v1 <= v3);
	BOOST_CHECK(v3 <= v2);
}


BOLT_AUTO_TEST_CASE(Arithmetics) {
	Int3 v1(1, 0, -1);
	Int3 result1(5, 0, -5);

	BOOST_CHECK_EQUAL(5 * v1, result1);

	Int3 v2(7, -3, 2);
	Int3 result2(8, -3, 1);

	BOOST_CHECK_EQUAL((v1 + v2), result2);
	BOOST_CHECK_EQUAL((result2 - v2), v1);

	Int3 v3 = v1;
	v3 += v2;
	BOOST_CHECK_EQUAL(v3, result2);
	v3 -= v1;
	BOOST_CHECK_EQUAL(v3, v2);

	BOOST_CHECK_EQUAL((Int3() + v1), v1);

	const Int3 result4{3, 2, 1};
	BOOST_CHECK_EQUAL((v1 + rep(2)), result4);
	BOOST_CHECK_EQUAL((rep(2) + v1), result4);
	BOOST_CHECK_EQUAL((result4 - rep(2)), v1);
	BOOST_CHECK_EQUAL((rep(-2) - v1), (-result4));
}


BOLT_AUTO_TEST_CASE(Algorithms) {
	BOOST_CHECK_EQUAL(dot(Int3(2, 4, 6), Int3(3, 0, -1)), 0);

	Int2 prod = product(Int2(2, 5), Int2(7, 3));
	BOOST_CHECK_EQUAL(prod[0], 14);
	BOOST_CHECK_EQUAL(prod[1], 15);

	Float4 v1(16.0f, 5.0f, -19.0f, -3.0f);
	Float4 v2(-1.0f, 15.0f, -4.2f, -0.01001f);
	BOOST_CHECK_EQUAL(sum(product(v1, v2)), dot(v1, v2));

	Int4 search_vector(1, 2, 3, 4);
	BOOST_CHECK_EQUAL(find(search_vector, 2), 1);
	BOOST_CHECK_EQUAL(find(search_vector, 4), 3);
	BOOST_CHECK_EQUAL(find(search_vector, 7), -1);
}


BOLT_AUTO_TEST_CASE(Division) {
	Int3 v(-1, -2, 4);
	Int3 d(8, 8, 8);

	BOOST_CHECK_EQUAL(Int3(7, 6, 4), modPeriodic(v, d));
	BOOST_CHECK_EQUAL(Int3(4, 4, 4), div(d, 2));
	BOOST_CHECK_EQUAL(Int3(4, 4, 4), div(d, Int3::fill(2)));
}


BOLT_AUTO_TEST_CASE(InsertRemoveDimension) {
	Int3 v3(1, 2, 3);
	Int2 v2_0(2, 3);
	Int2 v2_1(1, 3);
	Int2 v2_2(1, 2);

	BOOST_CHECK_EQUAL(removeDimension(v3, 0), v2_0);
	BOOST_CHECK_EQUAL(removeDimension(v3, 1), v2_1);
	BOOST_CHECK_EQUAL(removeDimension(v3, 2), v2_2);

	BOOST_CHECK_EQUAL(insertDimension(v2_0, 1, 0), v3);
	BOOST_CHECK_EQUAL(insertDimension(v2_1, 2, 1), v3);
	BOOST_CHECK_EQUAL(insertDimension(v2_2, 3, 2), v3);
}


BOLT_AUTO_TEST_CASE(RawAccess) {
	// Test that operator[] and direct access to the raw buffer array are equivalent
	Float4 v(7.3, -789, -165, 0.1651);

	for (int i = 0; i < 4; ++i) {
		BOOST_CHECK_EQUAL(v[i], v.pointer()[i]);
		BOOST_CHECK_EQUAL(&(v[i]), &(v.pointer()[i]));
	}
}

// NOTE(fidli): compilation errors
// Tests that array of vectors has linear memory layout
BOLT_AUTO_TEST_CASE(ArrayAccess, BOLT_TEST_SKIP) {
	struct SequenceGenerator {
		int current;
		SequenceGenerator() : current(-1) {}
		Float3 operator()() {
			++current;
			return Float3(
					current * 3,
					current * 3 + 1,
					current * 3 + 2);
		}
	};

	std::vector<Float3> sequence(100);
	// NOTE(fidli): this causes compilation error
	/*std::generate(begin(sequence), end(sequence), SequenceGenerator());
	float *ptr = reinterpret_cast<float *>(sequence.data());

	for (int i = 0; i < int(sequence.size() * 3); ++i) {
		BOOST_CHECK_EQUAL(sequence[i / 3][i % 3], ptr[i]);
	}
	*/
}


BOLT_AUTO_TEST_CASE(ArrayAlignment) {
	Vector<float, 5> array1[11];

	BOOST_CHECK_EQUAL(sizeof(array1), 11 * sizeof(Vector<float, 5>));
	BOOST_CHECK_EQUAL(11 * sizeof(Vector<float, 5>), 11 * 5 * sizeof(float));

	Vector<uint8_t, 7> array2[9];

	BOOST_CHECK_EQUAL(sizeof(array2), 9 * sizeof(Vector<uint8_t, 7>));
	BOOST_CHECK_EQUAL(9 * sizeof(Vector<uint8_t, 7>), 7 * 9 * sizeof(uint8_t));
}


}  // namespace bolt
