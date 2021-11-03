// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <boost/test/floating_point_comparison.hpp>

#include <string>

#include <boltview/host_image.h>
#include <boltview/reduce.h>
#include <boltview/copy.h>
#include <boltview/view_traits.h>
#include <boost/test/included/unit_test.hpp>
#include <boltview/tests/test_defs.h>

#if defined(__CUDACC__)
	#include <boltview/device_image.h>
	#include <boltview/device_image_view.h>
#endif  // __CUDACC__

namespace bolt {

static const double kFloatTestEpsilon = 0.000001;
static const float kFloatTestEpsilonSinglePrecision = kFloatTestEpsilon * 100;

template<typename TElement>
void checkElementsForEquality(TElement A, TElement B, float epsilon = kFloatTestEpsilon){
	// NOTE(fidli): boost_check_close uses Knuths tolerance https:// www.boost.org/doc/libs/1_65_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_impl.html
	// BOOST_CHECK_CLOSE does not work for one element being zero
	BOOST_CHECK_SMALL(A-B, epsilon);
}

template<typename TVector1, typename TVector2>
void testVectorsForIdentity(TVector1 v1, TVector2 v2, float epsilon = kFloatTestEpsilonSinglePrecision) {
	BOOST_CHECK_CLOSE(v1[0], v2[0], epsilon);
	BOOST_CHECK_CLOSE(v1[1], v2[1], epsilon);
	BOOST_CHECK_CLOSE(v1[2], v2[2], epsilon);

}

template<>
void checkElementsForEquality(DeviceComplexType A, DeviceComplexType B, float epsilon){
	// NOTE(fidli): boost_check_close uses Knuths tolerance https:// www.boost.org/doc/libs/1_65_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_impl.html
	// BOOST_CHECK_CLOSE does not work for one element being zero
	BOOST_CHECK_SMALL(A.x-B.x, epsilon);
	BOOST_CHECK_SMALL(A.y-B.y, epsilon);
}

template<>
void checkElementsForEquality(HostComplexType A, HostComplexType B, float epsilon){
	// NOTE(fidli): boost_check_close uses Knuths tolerance https:// www.boost.org/doc/libs/1_65_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_impl.html
	// BOOST_CHECK_CLOSE does not work for one element being zero
	BOOST_CHECK_SMALL(A.x-B.x, epsilon);
	BOOST_CHECK_SMALL(A.y-B.y, epsilon);
}

template<typename TView1, typename TView2, typename::std::enable_if<TView1::kIsHostView && TView2::kIsHostView && !(TView1::kIsDeviceView && TView2::kIsDeviceView)>::type * = nullptr>
void testViewsElementsForIdentity(TView1 view1, TView2 view2, float epsilon = kFloatTestEpsilon) {
	BOOST_CHECK_EQUAL(view1.size(), view2.size());
	auto host_view1 = view1;
	auto host_view2 = view2;
	for (int i = 0; i < host_view1.elementCount(); ++i) {
		auto A = linearAccess(host_view1, i);
		auto B = linearAccess(host_view2, i);
		checkElementsForEquality(A, B, epsilon);
	}
}

#if defined(__CUDACC__)
template<typename TView1, typename TView2, typename::std::enable_if<TView1::kIsDeviceView && TView2::kIsDeviceView>::type * = nullptr>
void testViewsElementsForIdentity(TView1 view1, TView2 view2, float epsilon = kFloatTestEpsilon) {
	BOOST_CHECK_EQUAL(view1.size(), view2.size());

	DeviceImage<typename TView1::Element, TView1::kDimension, typename TView1::Policy> device_image1(view1.size());
	DeviceImage<typename TView2::Element, TView1::kDimension, typename TView2::Policy> device_image2(view2.size());
	copy(view1, device_image1.view());
	copy(view2, device_image2.view());

	HostImage<typename TView1::Element, TView1::kDimension, typename TView1::Policy> host_image1(view1.size());
	HostImage<typename TView2::Element, TView1::kDimension, typename TView2::Policy> host_image2(view1.size());
	copy(device_image1.constView(), host_image1.view());
	copy(device_image2.constView(), host_image2.view());
	testViewsElementsForIdentity(host_image1.view(), host_image2.view());
}
#endif  // __CUDACC__


template<typename TView1, typename TView2>
float squareDifferenceOfImages(TView1 view1, TView2 view2) {
	return reduce(square(subtract(view1, view2)), 0.0f, thrust::plus<float>());
}

template<typename TView1, typename TView2>
void testViewsForIdentity(TView1 view1, TView2 view2, float epsilon = kFloatTestEpsilon) {
	float difference = squareDifferenceOfImages(view1, view2);
	BOOST_CHECK_SMALL(difference, epsilon);
	testViewsElementsForIdentity(view1, view2, epsilon);
}


template<typename TView>
void printDeviceView(TView view, std::ostream *stream) {
	DeviceImage<typename TView::Element, TView::kDimension, typename TView::Policy> device_image(view.size());
	copy(view, device_image.view());

	HostImage<typename TView::Element, TView::kDimension, typename TView::Policy> host_image(view.size());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (typename TView::TIndex k = 0; k < host_view.size()[2]; ++k) {
		for (typename TView::TIndex j = 0; j < host_view.size()[1]; ++j) {
			for (typename TView::TIndex i = 0; i < host_view.size()[0]; ++i) {
				(*stream) << std::setfill(' ') << std::setprecision(3) << std::fixed << std::setw(7) << host_view[Int3(i, j, k)] << ", ";
			}
			(*stream) << std::endl;
		}
		(*stream) << "--------------------------------------------\n";
	}
}


}  // namespace bolt
