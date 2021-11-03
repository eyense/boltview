#define BOOST_TEST_MODULE InterpolatedViewTest

#include <boltview/interpolated_view.h>


#include <boltview/copy.h>
#include <boltview/for_each.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <boltview/image_io.h>
#include <boltview/interpolation.h>
#include <boltview/procedural_views.h>
#include <boltview/view_iterators.h>
#include <boltview/reduce.h>
#include <boltview/tests/test_utils.h>

#include <boost/test/floating_point_comparison.hpp>
#include <boost/test/included/unit_test.hpp>


namespace bolt {


// constant value view interpolated, on host
BOLT_AUTO_TEST_CASE(ConstantHostViewInterpolationTest) {
	const Int3 value = Int3(1, 2, 3);
	HostImage<Int3, 2> host_image(8, 8);

	auto host_view = host_image.view();
	forEachPosition(host_view, [&](Int3& element, const Int2& index) { element = value; });

	auto interpolated_view = makeInterpolatedView<NearestNeighborInterpolator<BorderHandlingTraits<BorderHandling::kZero>>>(host_view);

	forEachPosition(host_view, [&](Int3& element, const Int2& index) {
		BOOST_CHECK_EQUAL(interpolated_view.access(Float2(index)), value);
		BOOST_CHECK_EQUAL(interpolated_view.access(Float2(sin(kPi * index[0] / 16) * 7, sin(kPi * index[1] / 16) * 7)), value);
	});
}


BOLT_AUTO_TEST_CASE(HostViewLinearInterpolationTest) {
	HostImage<float, 3> host_image(10, 10, 10);

	auto host_view = host_image.view();
	forEachPosition(host_view, [&](float& element, const Int3& index) { element = sum(index); });

	// linear interpolation view
	auto interpolated_view = makeInterpolatedView<LinearInterpolator<BorderHandlingTraits<BorderHandling::kNone>>>(host_view);

	forEachPosition(host_view, [&](float& element, const Int3& index) {
		BOOST_CHECK_EQUAL(interpolated_view.access(Float3(index)), sum(index));
		auto coords = Float3(sin(kPi * index[0] / 20) * 9, sin(kPi * index[1] / 20) * 9, sin(kPi * index[2] / 20) * 9);
		BOOST_CHECK_CLOSE(interpolated_view.access(coords), sum(coords), 0.0001);
	});
}

template<typename TView>
struct FloatPaddingFunctor {
	static const int kDimension = TView::kDimension;
	BOLT_DECL_HYBRID
	explicit FloatPaddingFunctor(TView view, Vector<float, kDimension> padding) : view_in_(view), padding_(padding) {}
	BOLT_DECL_HYBRID
	void operator()(typename TView::Element& element, const typename TView::IndexType& index) const {
		element = view_in_.access(index + padding_);
	}
	TView view_in_;
	Vector<float, kDimension> padding_;
};
template<typename TView>
FloatPaddingFunctor<TView> createFloatPaddingFunctor(TView view, Vector<float, TView::kDimension> padding) {
	return FloatPaddingFunctor<TView>(view, padding);
}

BOLT_AUTO_TEST_CASE(DeviceLinearInterpolationTest) {
	DeviceImage<float, 3> device_image(10, 10, 10);
	DeviceImage<float, 3> padded_image(10, 10, 10);
	HostImage<float, 3> host_image(10, 10, 10);
	HostImage<float, 3> control_image(10, 10, 10);

	// filling the data on host
	auto host_view = host_image.view();
	for (auto pair : zipWithPosition(host_view)) {
		pair.element = sum(pair.position);
	}

	auto device_view = device_image.view();
	// copying the data to device
	copy(host_view, device_view);
	// device linear interpolation view with mirror border
	auto interpolated_view = makeInterpolatedView<LinearInterpolator<BorderHandlingTraits<BorderHandling::kMirror>>>(device_view);

	auto padding = Float3(0.6f, 1.4f, 2.8f);
	auto padding_functor = createFloatPaddingFunctor(interpolated_view, padding);
	auto padded_view = padded_image.view();
	// creating device padded view, using padding_functor and foreach to copy values to padded view
	forEachPosition(padded_view, padding_functor);
	auto control_view = control_image.view();
	// copying the filled padded view to host - control_view
	copy(padded_view, control_view);

	// checking the correct values in the control_view
	// the coordinate sum does not match the interpolated values outside the original boundaries, due to the influence of border
	for (int i = 0; i < int(10 - padding[0]); ++i) {
		for (int j = 0; j < int(10 - padding[1]); ++j) {
			for (int k = 0; k < int(10 - padding[2]); ++k) {
				BOOST_CHECK_CLOSE(control_view[Int3(i, j, k)], i + j + k + sum(padding), 0.0001);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(HostTricubicInterpolationTest) {
	HostImage<float, 3> host_image(10, 10, 10);

	auto host_view = host_image.view();
	forEachPosition(host_view, [&](float& element, const Int3& index) { element = sum(index); });

	// cubic interpolation view
	auto interpolated_view = makeInterpolatedView<CubicInterpolator<BorderHandlingTraits<BorderHandling::kRepeat>>>(host_view);

	// the coordinate sum does not match the interpolated values near the borders due to the influence of borders
	for (int i = 1; i < 8; ++i) {
		for (int j = 1; j < 8; ++j) {
			for (int k = 1; k < 8; ++k) {
				BOOST_CHECK_EQUAL(interpolated_view.access(Float3(i, j, k)), i + j + k);
				auto coords = Float3(i + 0.5f, j + 0.5f, k + 0.5f);
				BOOST_CHECK_CLOSE(interpolated_view.access(coords), sum(coords), 0.0001f);
				coords = Float3(sin(kPi * i / 20) * 8, sin(kPi * j / 20) * 8, sin(kPi * k / 20) * 8);
				BOOST_CHECK_CLOSE(interpolated_view.access(coords), sum(coords), 0.0001f);
			}
		}
	}
}

BOLT_AUTO_TEST_CASE(DeviceTricubicInterpolationTest) {
	DeviceImage<Float3, 3> device_image(20, 20, 20);
	DeviceImage<Float3, 3> padded_image(20, 20, 20);
	HostImage<Float3, 3> host_image(20, 20, 20);
	HostImage<Float3, 3> control_image(20, 20, 20);

	// filling the data on host
	auto host_view = host_image.view();
	for (auto pair : zipWithPosition(host_view)) {
		pair.element = Float3(
			sum(pair.position), pair.position[0] + pair.position[1] - pair.position[2], pair.position[0] - pair.position[1] - pair.position[2]);
	}

	auto device_view = device_image.view();
	// copying the data to device
	copy(host_view, device_view);
	// device cubic interpolation view with mirror border
	auto interpolated_view = makeInterpolatedView<CubicInterpolator<BorderHandlingTraits<BorderHandling::kMirror>>>(device_view);

	auto padding = Float3(0.6f, 1.4f, 2.8f);
	auto padding_functor = createFloatPaddingFunctor(interpolated_view, padding);
	auto padded_view = padded_image.view();
	// creating device padded view, using padding_functor and foreach
	forEachPosition(padded_view, padding_functor);
	auto control_view = control_image.view();
	// copying the filled padded view to host - control_view
	copy(padded_view, control_view);

	// checking the correct values in the control_view
	// the coordinate sum does not match the interpolated values when the interpolator accesses values outside the box
	for (int i = 1; i < int(20 - padding[0] - 1); ++i) {
		for (int j = 1; j < int(20 - padding[1] - 1); ++j) {
			for (int k = 1; k < int(20 - padding[2] - 1); ++k) {
				BOOST_CHECK_CLOSE(control_view[Int3(i, j, k)][0], i + j + k + sum(padding), 0.001);
				BOOST_CHECK_CLOSE(control_view[Int3(i, j, k)][1], i + j - k + sum(padding) - 2 * padding[2], 0.001);
				BOOST_CHECK_CLOSE(control_view[Int3(i, j, k)][2], i - j - k - sum(padding) + 2 * padding[0], 0.001);
			}
		}
	}
}

struct AddFunctor {
  	BOLT_DECL_HYBRID
	float operator()(float val1, float val2) const {
		return val1 + val2;
	}
};

BOLT_AUTO_TEST_CASE(DeviceInterpolationBoudedAccessTest) {
	HostImage<float, 2> host_image(10, 10);
	DeviceImage<float, 2> device_image(10, 10);
	HostImage<float, 2> host_image2(1, 1);
	DeviceImage<float, 2> device_image2(1, 1);

	// filling the data on host
	auto host_view = host_image.view();
	for (auto pair : zipWithPosition(host_view)) {
		pair.element = sum(pair.position);
	}
	auto device_view = device_image.view();
	copy(host_view, device_view);

	auto host_view2 = host_image2.view();
	host_view2[Int2{0,0}] = 1;
	auto device_view2 = device_image2.view();
	copy(host_view2, device_view2);

	auto interpolated_view = makeInterpolatedView<LinearInterpolator<BorderHandlingTraits<BorderHandling::kRepeat>>>(device_view2);

	auto sum_view = nAryOperator(AddFunctor{}, device_view, interpolated_view);

  	ExecutionPolicy policy{};
	auto result = sum(sum_view, policy);

  	BOOST_CHECK_EQUAL(result, sum(device_view, policy)+product(device_view.size()));
}

}  // namespace bolt
