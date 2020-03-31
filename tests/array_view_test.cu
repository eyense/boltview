// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#define BOOST_TEST_MODULE ArrayViewTest
#include <boost/test/included/unit_test.hpp>
#include "tests/test_utils.h"

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <boltview/copy.h>
#include <boltview/array_view.h>

namespace bolt {

BOLT_AUTO_TEST_CASE(HostArrayConstViewTest) {
	thrust::host_vector<int> test_vector(10);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeHostArrayConstView(test_vector);

	BOOST_CHECK_EQUAL(view.size(), test_vector.size());
	for (int i = 0; i < static_cast<int>(test_vector.size()); ++i) {
		BOOST_CHECK_EQUAL(test_vector[i], view[i]);
	}
}

BOLT_AUTO_TEST_CASE(HostArrayViewTest) {
	thrust::host_vector<int> test_vector(10);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeHostArrayView(test_vector);
	BOOST_CHECK_EQUAL(view.size(), test_vector.size());

	for (int i = 0; i < static_cast<int>(test_vector.size()); ++i) {
		BOOST_CHECK_EQUAL(test_vector[i], view[i]);
	}
}

template<typename TArrayView>
BOLT_GLOBAL void ArrayTestKernel(
		int *test_data,
		TArrayView array_view,
		int *result,
		int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		result[index] = test_data[index] - array_view[index];
	}
}


BOLT_AUTO_TEST_CASE(DeviceArrayViewTest) {
	thrust::device_vector<int> test_vector(1000);
	thrust::device_vector<int> result(1000);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeDeviceArrayView(test_vector);
	BOOST_CHECK_EQUAL(view.size(), test_vector.size());

	ArrayTestKernel<decltype(view)><<<10, 128>>>(
		thrust::raw_pointer_cast(test_vector.data()),
		view,
		thrust::raw_pointer_cast(result.data()),
		test_vector.size());

	BOLT_CHECK(cudaThreadSynchronize());

	thrust::host_vector<int> host_result = result;
	for (int i = 0; i < static_cast<int>(result.size()); ++i) {
		BOOST_CHECK_EQUAL(result[i], 0);
	}
}

BOLT_AUTO_TEST_CASE(DeviceArrayConstViewTest) {
	thrust::device_vector<int> test_vector(1000);
	thrust::device_vector<int> result(1000);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeDeviceArrayConstView(test_vector);
	BOOST_CHECK_EQUAL(view.size(), test_vector.size());

	ArrayTestKernel<decltype(view)><<<10, 128>>>(
		thrust::raw_pointer_cast(test_vector.data()),
		view,
		thrust::raw_pointer_cast(result.data()),
		test_vector.size());

	BOLT_CHECK(cudaThreadSynchronize());

	thrust::host_vector<int> host_result = result;
	for (int i = 0; i < static_cast<int>(result.size()); ++i) {
		BOOST_CHECK_EQUAL(result[i], 0);
	}
}

BOLT_AUTO_TEST_CASE(HostArraySubviews) {
	thrust::host_vector<int> test_vector(1000);
	thrust::sequence(test_vector.begin(), test_vector.end());

	auto view = makeArrayView(test_vector);

	auto array_subview = subview(view, 100, 200);

	BOOST_CHECK_EQUAL(array_subview.size(), 100);
	for (int i = 0; i < static_cast<int>(array_subview.size()); ++i) {
		BOOST_CHECK_EQUAL(array_subview[i], 100 + i);
	}
}

BOOST_AUTO_TEST_CASE(HostToHostCopy){

	thrust::host_vector<int> test_vector(1000);
	auto view1 = makeArrayView(test_vector);

	thrust::sequence(test_vector.begin(), test_vector.end());
	auto view2 = makeArrayView(test_vector);
	copy(view2, view1);
	for(int i = 0; i < static_cast<int>(test_vector.size()); i++){
		BOOST_CHECK_EQUAL(view1[i], view2[i]);
	}

}

}  // namespace bolt
