#define BOOST_TEST_MODULE ViewIteratorsTest

#include <boltview/view_iterators.h>

#include <boltview/copy.h>
#include <boltview/for_each.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/image_io.h>
#include <boltview/procedural_views.h>
#include <boltview/array_view.h>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boltview/tests/test_utils.h>


namespace bolt {


BOLT_AUTO_TEST_CASE(IteratorTest) {
	Int3 value = Int3(1, 2, 3);
	HostImage<Int3, 2> host_image(8, 8);
	auto host_view = host_image.view();
	for (auto& access : host_view) {
		access = value;
	}

	for (auto elem : host_view) {
		BOOST_CHECK_EQUAL(elem, value);
	}
}

BOLT_AUTO_TEST_CASE(IteratorTestLongInt) {
	Int3 value = Int3(1, 2, 3);
	HostImage<Int3, 2, LongIndexViewPolicy> host_image(8, 8);
	auto host_view = host_image.view();
	for (auto& access : host_view) {
		access = value;
	}

	for (auto elem : host_view) {
		BOOST_CHECK_EQUAL(elem, value);
	}
}

BOLT_AUTO_TEST_CASE(ZipTest) {
	HostImage<int, 2> host_image(8, 8);
	auto host_view = host_image.view();

	for (auto pair : zipWithPosition(host_view)) {
		pair.element = sum(pair.position);
	}

	for (auto pair : zipWithPosition(host_view)) {
		BOOST_CHECK_EQUAL(pair.element, sum(pair.position));
	}
}

BOLT_AUTO_TEST_CASE(ZipTestLongInt) {
	HostImage<int, 2, LongIndexViewPolicy> host_image(8, 8);
	auto host_view = host_image.view();

	for (auto pair : zipWithPosition(host_view)) {
		pair.element = sum(pair.position);
	}

	for (auto pair : zipWithPosition(host_view)) {
		BOOST_CHECK_EQUAL(pair.element, sum(pair.position));
	}
}

BOLT_AUTO_TEST_CASE(ConstViewIteratorTest) {
	thrust::host_vector<int> test_vector(10);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeHostArrayConstView(test_vector);

	int i = 0;
	for (auto element : view) {
		BOOST_CHECK_EQUAL(element, i);
		i++;
	}
}


BOLT_AUTO_TEST_CASE(ReverseIteratorTest) {
	HostImage<int, 2> host_image(8, 8);
	auto host_view = host_image.view();
	auto view = zipWithPosition(host_view);

	for (auto it = begin(view); it != end(view); ++it) {
		(*it).element = sum((*it).position);
	}

	for (auto pair : zipWithPosition(host_view)) {
		BOOST_CHECK_EQUAL(pair.element, sum(pair.position));
	}
}

BOLT_AUTO_TEST_CASE(ReverseIteratorTestLongInt) {
	HostImage<int, 2, LongIndexViewPolicy> host_image(8, 8);
	auto host_view = host_image.view();
	auto view = zipWithPosition(host_view);

	for (auto it = begin(view); it != end(view); ++it) {
		(*it).element = sum((*it).position);
	}

	for (auto pair : zipWithPosition(host_view)) {
		BOOST_CHECK_EQUAL(pair.element, sum(pair.position));
	}
}

BOLT_AUTO_TEST_CASE(VectorIteratorTest) {
	int value = 5;
	Vector<int, 10> vector;
	for (auto& element : vector) {
		element = value;
	}

	const auto copy = vector;

	for (auto element : vector) {
		BOOST_CHECK_EQUAL(element, value);
	}

	for (auto element : copy) {
		BOOST_CHECK_EQUAL(element, value);
	}
}

BOLT_AUTO_TEST_CASE(VectorIteratorOnDeviceTest) {
	DeviceImage<Vector<int, 10>, 3> device_image(10, 10, 10);
	HostImage<Vector<int, 10>, 3> host_image(10, 10, 10);

	auto device_view = device_image.view();
	forEachPosition(device_view, [] __device__(Vector<int, 10> & vector_element, const Int3& index) {
		for (auto& element : vector_element) {
			element = sum(index);
		}
	});

	auto host_view = host_image.view();
	copy(device_view, host_view);

	for (auto pair : zipWithPosition(host_view)) {
		for (auto elem : pair.element) {
			BOOST_CHECK_EQUAL(elem, sum(pair.position));
		}
	}
}


/*
// Does not compile
BOLT_AUTO_TEST_CASE(VectorIteratorOnDeviceTestLongInt) {
	DeviceImage<Vector<int, 10>, 3, LongIndexViewPolicy> device_image(10, 10, 10);
	HostImage<Vector<int, 10>, 3, LongIndexViewPolicy> host_image(10, 10, 10);

	auto device_view = device_image.view();
	forEachPosition(device_view, [] __device__(Vector<int, 10> & vector_element, const LongInt3& index) {
		for (auto& element : vector_element) {
			element = sum(index);
		}
	});

	auto host_view = host_image.view();
	copy(device_view, host_view);

	for (auto pair : zipWithPosition(host_view)) {
		for (auto elem : pair.element) {
			BOOST_CHECK_EQUAL(elem, sum(pair.position));
		}
	}
}
*/

}  // namespace bolt
