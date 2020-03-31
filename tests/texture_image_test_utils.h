// Copyright 2016 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <boltview/copy.h>
#include <boltview/host_image.h>
#include <boltview/device_image.h>
#include <tests/test_utils.h>
#include <algorithm>
#include <boost/test/test_tools.hpp>
#include <limits>
#include <random>
#include <vector>

namespace bolt {

template<typename TType>
struct DistributionType {};

template<>
struct DistributionType<int> {
	typedef std::uniform_int_distribution<int> type;
	static int DefaultStart() {
		return 0;
	}
	static int DefaultEnd() {
		return 255;
	}
};

template<>
struct DistributionType<float> {
	typedef std::uniform_real_distribution<float> type;
	static float DefaultStart() {
		return 0.0f;
	}
	static float DefaultEnd() {
		return 1.0f;
	}
};

template<typename TType>
class RandomGenerator {
public:
	explicit RandomGenerator(uint64_t seed) :
		_random_engine(seed),
		_distribution(DistributionType<TType>::DefaultStart(), DistributionType<TType>::DefaultEnd()) {}
	TType operator()() {
		return _distribution(_random_engine);
	}
private:
	typename DistributionType<TType>::type _distribution;
	std::default_random_engine _random_engine;
};

template<typename TType, int tDimension>
class RandomGenerator<Vector<TType, tDimension>> {
public:
	explicit RandomGenerator(uint64_t seed) :
		_random_engine(seed),
		_distribution(DistributionType<TType>::DefaultStart(), DistributionType<TType>::DefaultEnd()) {}
	Vector<TType, tDimension> operator()() {
		Vector<TType, tDimension> random_vector;
		for (int i = 0; i < tDimension; i++) {
			random_vector[i] = _distribution(_random_engine);
		}
		return random_vector;
	}
private:
	typename DistributionType<TType>::type _distribution;
	std::default_random_engine _random_engine;
};

template<typename TView>
void generateRandomView(TView view, uint64_t seed) {
	RandomGenerator<typename TView::Element> random_generator(seed);
	for(int i = 0; i < view.elementCount(); i++){
		linearAccess(view, i) = random_generator();
	}
}

template<typename TType>
void CheckElementCloseToZero(const TType& element) {
	BOOST_CHECK(std::abs(element) <= std::numeric_limits<TType>::epsilon());
}

template<typename TType, int tDimension>
void CheckElementCloseToZero(const Vector<TType, tDimension>& element) {
	for(int i = 0; i < tDimension; i++) {
		BOOST_CHECK(std::abs(element[i]) <= std::numeric_limits<TType>::epsilon());
	}
}

template<typename TDeviceView>
void CheckViewCloseToZero(TDeviceView device_view) {
	using TElement = typename TDeviceView::Element;
	DeviceImage<TElement, TDeviceView::kDimension> device_image(device_view.size());
	copy(device_view, device_image.view());

	HostImage<TElement, TDeviceView::kDimension> host_image(device_view.size());
	copy(device_image.view(), host_image.view());

	auto host_view = host_image.view();
	for (int i = 0; i < host_view.elementCount(); i++) {
		checkElementCloseToZero(linearAccess(host_view, i));
	}
}

template<typename TView>
void PrintDeviceView(TView view) {
	DeviceImage<float, 2> device_image(view.size());
	copy(view, device_image.view());

	HostImage<float, 2> host_image(view.size());
	copy(device_image.constView(), host_image.view());

	auto host_view = host_image.constView();
	for (int k = 0; k < host_view.size()[1]; ++k) {
		for (int j = 0; j < host_view.size()[0]; ++j) {
			std::cout << std::setfill(' ') << std::setprecision(3) << std::fixed << std::setw(7) << host_view[Int2(j, k)] << ", ";
		}
		std::cout << "--------------------------------------------\n";
	}
}

}  // namespace bolt
