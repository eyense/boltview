// Copyright 2016 Eyen SE
// Author: Lukas Marsalek, lukas.marsalek@eyen.se


#define BOOST_TEST_MODULE GilViewAdaptorTest
#include <boltview/tests/test_utils.h>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <initializer_list>

#include <boost/test/included/unit_test.hpp>

#include <boltview/host_image_view.h>
#include <boltview/device_image.h>
#include <boltview/copy.h>
#include <boltview/gil_adaptor_image.h>

namespace bolt {

namespace bg = boost::gil;

BOLT_AUTO_TEST_CASE(ElementTypeSelection){

	static_assert(std::is_integral<GilAdaptorImage<bg::gray8_view_t>::Element>::value,
			"Element type of GilAdaptorView constructed from Gil view with integral pixel type is not an integer");
	static_assert(std::is_floating_point<GilAdaptorImage<bg::gray32f_view_t>::Element>::value,
			"Element type of GilAdaptorView constructed from Gil view with scoped_float pixel type is not float");
}

template<typename GilViewType>
void HostDeviceCopyTestBody() {
	typedef GilAdaptorImage<GilViewType> Adaptor_type;

	const unsigned int width = 8;
	const unsigned int height = 8;

	Adaptor_type frame(width, height);

	bolt::DeviceImage<typename Adaptor_type::Element, 2> frame_device(frame.size());
	bolt::copy(frame.hostImageView(), frame_device.view());
}

template<typename ... Types> void copyTestLoopOverTypes(){
	std::initializer_list<int> dummyList = {(HostDeviceCopyTestBody<Types>(), 0)...};
}

BOLT_AUTO_TEST_CASE(HostDeviceCopyAllTypes){

	copyTestLoopOverTypes<
		bg::gray8_view_t,
		bg::gray8s_view_t,
		bg::gray16_view_t,
		bg::gray16s_view_t,
		bg::gray32_view_t,
		bg::gray32s_view_t,
		bg::gray32f_view_t,
		bg::bgr8_view_t,
		bg::bgr8s_view_t,
		bg::bgr16_view_t,
		bg::bgr16s_view_t,
		bg::bgr32_view_t,
		bg::bgr32s_view_t,
		bg::bgr32f_view_t,
		bg::argb8_view_t,
		bg::argb8s_view_t,
		bg::argb16_view_t,
		bg::argb16s_view_t,
		bg::argb32_view_t,
		bg::argb32s_view_t,
		bg::argb32f_view_t,
		bg::abgr8_view_t,
		bg::abgr8s_view_t,
		bg::abgr16_view_t,
		bg::abgr16s_view_t,
		bg::abgr32_view_t,
		bg::abgr32s_view_t,
		bg::abgr32f_view_t,
		bg::bgra8_view_t,
		bg::bgra8s_view_t,
		bg::bgra16_view_t,
		bg::bgra16s_view_t,
		bg::bgra32_view_t,
		bg::bgra32s_view_t,
		bg::bgra32f_view_t,
		bg::rgb8_view_t,
		bg::rgb8s_view_t,
		bg::rgb16_view_t,
		bg::rgb16s_view_t,
		bg::rgb32_view_t,
		bg::rgb32s_view_t,
		bg::rgb32f_view_t,
		bg::rgba8_view_t,
		bg::rgba8s_view_t,
		bg::rgba16_view_t,
		bg::rgba16s_view_t,
		bg::rgba32_view_t,
		bg::rgba32s_view_t,
		bg::rgba32f_view_t,
		bg::cmyk8_view_t,
		bg::cmyk8s_view_t,
		bg::cmyk16_view_t,
		bg::cmyk16s_view_t,
		bg::cmyk32_view_t,
		bg::cmyk32s_view_t,
		bg::cmyk32f_view_t>();
}

}  // namespace bolt
