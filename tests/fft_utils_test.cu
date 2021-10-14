// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se


#define BOOST_TEST_MODULE FftUtilsTest
#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <complex>
#include <cufft.h>

#include <iostream>
#include <iomanip>


#include <boltview/device_image.h>
#include <boltview/device_image_view.h>
#include <boltview/host_image.h>
#include "tests/test_utils.h"

#include <boltview/math/complex.h>
#include <boltview/fft/fft_calculator.h>
#include <boltview/fft/fft_utils.h>
#include <boltview/fft/fft_views.h>

#include <boltview/image_io.h>

#include <boltview/subview.h>

namespace bolt {

static const double kFftEpsilon = 0.0001;

BOOST_AUTO_TEST_CASE(ComplexOperators){
	float a = 3;
	float b = 2;
	float c = 4;
	float d = -3;
	// device
	DeviceComplexType devA = {a, b};
	DeviceComplexType devB = {c, d};
	// host
	HostComplexType hostA = {a, b};
	HostComplexType hostB = {c, d};

	// binary

	// operator +
	{
		{ // device
			DeviceComplexType r = devA + devB;
			BOOST_CHECK_CLOSE(r.x, a+c, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, b+d, kFftEpsilon);
			r = devB + devA;
			BOOST_CHECK_CLOSE(r.x, c+a, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, d+b, kFftEpsilon);
		}
		{ // host
			HostComplexType r = hostA + hostB;
			BOOST_CHECK_CLOSE(r.x, a+c, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, b+d, kFftEpsilon);
			r = hostB + hostA;
			BOOST_CHECK_CLOSE(r.x, c+a, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, d+b, kFftEpsilon);
		}
	}

	// operator -
	{
		{// device
			DeviceComplexType r = devA - devB;
			BOOST_CHECK_CLOSE(r.x, a-c, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, b-d, kFftEpsilon);
			r = devB - devA;
			BOOST_CHECK_CLOSE(r.x, c-a, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, d-b, kFftEpsilon);
		}
		{// host
			HostComplexType r = hostA - hostB;
			BOOST_CHECK_CLOSE(r.x, a-c, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, b-d, kFftEpsilon);
			r = hostB - hostA;
			BOOST_CHECK_CLOSE(r.x, c-a, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, d-b, kFftEpsilon);
		}
	}

	// operator /
	{
		{// device
			DeviceComplexType r = devA / devB;
			BOOST_CHECK_CLOSE(r.x, 6.0f/25.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 17.0f/25.0f, kFftEpsilon);
			r = devB / devA;
			BOOST_CHECK_CLOSE(r.x, 6.0f/13.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -17.0f/13.0f, kFftEpsilon);

			r = devA / 3.0f;
			BOOST_CHECK_CLOSE(r.x, 3.0f/3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f/3.0f, kFftEpsilon);
		}
		{// host
			HostComplexType r = hostA / hostB;
			BOOST_CHECK_CLOSE(r.x, 6.0f/25.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 17.0f/25.0f, kFftEpsilon);
			r = hostB / hostA;
			BOOST_CHECK_CLOSE(r.x, 6.0f/13.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -17.0f/13.0f, kFftEpsilon);

			r = hostA / 3.0f;
			BOOST_CHECK_CLOSE(r.x, 3.0f/3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f/3.0f, kFftEpsilon);
			}

	}

	// operator *
	{
		{// device
			DeviceComplexType r = devA * devB;
			BOOST_CHECK_CLOSE(r.x, 18.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -1.0f, kFftEpsilon);
			r = devB * devA;
			BOOST_CHECK_CLOSE(r.x, 18.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -1.0f, kFftEpsilon);


			r = devA * 3.0f;
			BOOST_CHECK_CLOSE(r.x, 3.0f*3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f*3.0f, kFftEpsilon);
			r = 3.0f * devA;
			BOOST_CHECK_CLOSE(r.x, 3.0f*3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f*3.0f, kFftEpsilon);
		}
		{// host
			HostComplexType r = hostA * hostB;
			BOOST_CHECK_CLOSE(r.x, 18.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -1.0f, kFftEpsilon);
			r = hostB * hostA;
			BOOST_CHECK_CLOSE(r.x, 18.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, -1.0f, kFftEpsilon);


			r = hostA * 3.0f;
			BOOST_CHECK_CLOSE(r.x, 3.0f*3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f*3.0f, kFftEpsilon);
			r = 3.0f * hostA;
			BOOST_CHECK_CLOSE(r.x, 3.0f*3.0f, kFftEpsilon);
			BOOST_CHECK_CLOSE(r.y, 2.0f*3.0f, kFftEpsilon);
		}
	}

	// operations
	{
		// NOTE(fidli): these could be randomized
		float mag = 0.658342f;
		float ph = 0.7640132f * kPi;
		// magnitude
		{
			{// device
				DeviceComplexType r = makeComplex<DeviceComplexType>(mag, ph);
				BOOST_CHECK_SMALL((magnitude(r) - mag), (static_cast<float>(kFftEpsilon)));
				BOOST_CHECK_SMALL((phase(r) - ph), (static_cast<float>(kFftEpsilon)));
			}
			{// host
				HostComplexType r = makeComplex<HostComplexType>(mag, ph);
				BOOST_CHECK_SMALL((magnitude(r) - mag), (static_cast<float>(kFftEpsilon)));
				BOOST_CHECK_SMALL((phase(r) - ph), (static_cast<float>(kFftEpsilon)));
			}
		}
		// phase
		{
			{// device
				DeviceComplexType r = makeComplex<DeviceComplexType>(-mag, -ph);
				BOOST_CHECK_SMALL((magnitude(r) - mag), (static_cast<float>(kFftEpsilon)));
				BOOST_CHECK_SMALL((phase(r) + static_cast<float>(kPi) - (-ph+(2*static_cast<float>(kPi)))), (static_cast<float>(kFftEpsilon)));
			}
			{// host
				HostComplexType r = makeComplex<HostComplexType>(-mag, -ph);

				BOOST_CHECK_SMALL((magnitude(r) - mag), (static_cast<float>(kFftEpsilon)));
				BOOST_CHECK_SMALL((phase(r) + static_cast<float>(kPi)  - (-ph+(2*static_cast<float>(kPi)))), (static_cast<float>(kFftEpsilon)));
			}
		}
	}
}

template<int tDim>
void unaryOperationsTestOnSine(Vector<int, tDim> size, Vector<float, tDim> amplitude, Vector<float, tDim> frequency, Vector<float, tDim> ph){
	auto sinus_view = makeSinusImageView(size, amplitude, frequency, ph);

	FftCalculator<tDim,DeviceFftPolicy<Forward>> forward(size);

	auto image = forward.createSpaceDomainDeviceImage();
	copy(sinus_view, image.view());
	auto image_fft = forward.createFrequencyDomainDeviceImage();

	DeviceImage<float, tDim> devicePhase(image_fft.size());
	DeviceImage<float, tDim> deviceMagnitude(image_fft.size());
	DeviceImage<cufftComplex, tDim> deviceConjugate(image_fft.size());

	auto hostFrequency = forward.createFrequencyDomainHostImage();
	HostImage<float, tDim> devicePhaseResult(image_fft.size());
	HostImage<float, tDim> deviceMagnitudeResult(image_fft.size());
	HostImage<cufftComplex, tDim> deviceConjugateResult(image_fft.size());

	forward.calculate(image.view(), image_fft.view());
	copy(multiplyByFactor(1.0f / sinus_view.elementCount(), image_fft.constView()), image_fft.view());

	copy(phase(image_fft.view()), devicePhase.view());
	copy(magnitude(image_fft.view()), deviceMagnitude.view());
	copy(conjugate(image_fft.view()), deviceConjugate.view());
	copy(image_fft.view(), hostFrequency.view());
	copy(devicePhase.view(), devicePhaseResult.view());
	copy(deviceMagnitude.view(), deviceMagnitudeResult.view());
	copy(deviceConjugate.view(), deviceConjugateResult.view());

	auto hostPhaseView = phase(hostFrequency.view());
	auto hostMagnitudeView = magnitude(hostFrequency.view());
	auto hostConjugateView = conjugate(hostFrequency.view());

	for(int i = 0; i < product(image_fft.size()); i++){
		// phase
		{
			auto device = linearAccess(devicePhaseResult.view(), i);
			auto host = linearAccess(hostPhaseView, i);
			BOOST_CHECK_SMALL(device-host, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(host-phase(linearAccess(hostFrequency.view(), i)), static_cast<float>(kFftEpsilon));
		}
		// magnitude
		{
			auto device = linearAccess(deviceMagnitudeResult.view(), i);
			auto host = linearAccess(hostMagnitudeView, i);
			BOOST_CHECK_SMALL(device-host, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(host-magnitude(linearAccess(hostFrequency.view(), i)), static_cast<float>(kFftEpsilon));
		}
		// conjugate
		{
			auto device = linearAccess(deviceConjugateResult.view(), i);
			auto host = linearAccess(hostConjugateView, i);
			auto devHost = device - host;
			auto hostProc = host-conjugate(linearAccess(hostFrequency.view(), i));

			BOOST_CHECK_SMALL(devHost.x, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(devHost.y, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(hostProc.x, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(hostProc.y, static_cast<float>(kFftEpsilon));
		}
	}
}

BOOST_AUTO_TEST_CASE(UnaryOperationsViews){
	unaryOperationsTestOnSine(Int1(32), Float1(1), Float1(1), Float1());
	unaryOperationsTestOnSine(Int2(32, 32), Float2(1,1), Float2(1,1), Float2());
	unaryOperationsTestOnSine(Int3(32, 32, 32), Float3(1,1,1), Float3(1,1,1), Float3());
}
template<int tDim>
void BinaryOperationsTestOnSines(Vector<int, tDim> size, Vector<float, tDim> amplitudeA, Vector<float, tDim> amplitudeB, Vector<float, tDim> frequencyA, Vector<float, tDim> frequencyB, Vector<float, tDim> phaseA, Vector<float, tDim> phaseB){
	auto viewA = makeSinusImageView(size, amplitudeA, frequencyA, phaseA);
	auto viewB = makeSinusImageView(size, amplitudeB, frequencyB, phaseB);

	FftCalculator<tDim,DeviceFftPolicy<Forward>> forward(viewA.size());

	auto imageA = forward.createSpaceDomainDeviceImage();
	auto imageB = forward.createSpaceDomainDeviceImage();
	copy(viewA, imageA.view());
	copy(viewB, imageB.view());
	auto image_fftA = forward.createFrequencyDomainDeviceImage();
	auto image_fftB = forward.createFrequencyDomainDeviceImage();

	forward.calculate(imageA.view(), image_fftA.view());
	copy(multiplyByFactor(1.0f / viewA.elementCount(), image_fftA.constView()), image_fftA.view());
	forward.calculate(imageB.view(), image_fftB.view());
	copy(multiplyByFactor(1.0f / viewB.elementCount(), image_fftB.constView()), image_fftB.view());


	auto hostFrequencyA = forward.createFrequencyDomainHostImage();
	auto hostFrequencyB = forward.createFrequencyDomainHostImage();
	copy(image_fftA.view(), hostFrequencyA.view());
	copy(image_fftB.view(), hostFrequencyB.view());

	auto deviceHadamard = forward.createFrequencyDomainDeviceImage();
	auto deviceHadamardResult = forward.createFrequencyDomainHostImage();

	copy(hadamard(image_fftA.view(), image_fftB.view()), deviceHadamard.view());
	copy(deviceHadamard.view(), deviceHadamardResult.view());

	auto hostHadamardView = hadamard(hostFrequencyA.view(), hostFrequencyB.view());

	for(int i = 0; i < product(image_fftA.size()); i++){
		// hadamard
		{
			auto device = linearAccess(deviceHadamardResult.view(), i);
			auto host = linearAccess(hostHadamardView, i);
			auto devHost = device-host;
			auto hostProc = host-hadamard(linearAccess(hostFrequencyA.view(), i), linearAccess(hostFrequencyB.view(), i));
			BOOST_CHECK_SMALL(devHost.x, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(devHost.y, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(hostProc.x, static_cast<float>(kFftEpsilon));
			BOOST_CHECK_SMALL(hostProc.y, static_cast<float>(kFftEpsilon));
		}
	}

}

BOOST_AUTO_TEST_CASE(BinaryOperationsViews){
	BinaryOperationsTestOnSines(Int1(32), Float1(5), Float1(1), Float1(1), Float1(1.5f),  Float1(), Float1(kPi));
	BinaryOperationsTestOnSines(Int2(32, 32), Float2(5,1), Float2(1,2), Float2(1,0.2f), Float2(1.5f,1),  Float2(), Float2(kPi, 0));
	BinaryOperationsTestOnSines(Int3(32, 32, 32), Float3(5,1,3), Float3(1,2,8),  Float3(1,0.2f,1.41f), Float3(1.5f,1,3.333f),  Float3(), Float3(kPi, 0, 0.34));
}


BOOST_AUTO_TEST_CASE(OnlyDCComponent){

	float constantValue = 1.0f;
	auto size = Int3(32, 32, 32);
	auto fftSize = getFftImageSize(size);

	auto view = makeConstantImageView(constantValue, size);
	HostImage<float, 3> device_check(fftSize);
	HostImage<float, 3> host_check(fftSize);

	{// device
		FftCalculator<3,DeviceFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainDeviceImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainDeviceImage();

		forward.calculateAsync(image.view(), image_fft.view());
		BOLT_CHECK(cudaThreadSynchronize());


		DeviceImage<float, 3> amplitude_image(image_fft.size());
		HostImage<float, 3> amplitude_host_image(image_fft.size());
		copy(amplitude(image_fft.constView()), amplitude_image.view());
		copy(amplitude_image.constView(), device_check.view());
	}

	{// host
		FftCalculator<3,HostFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainHostImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainHostImage();

		forward.calculate(image.view(), image_fft.view());

		copy(amplitude(image_fft.constView()), host_check.view());
	}

	// check
	auto host_check_view = host_check.constView();
	auto device_check_view = device_check.constView();
	for (int k = 0; k < fftSize[2]; ++k) {
		for (int j = 0; j <  fftSize[1]; ++j) {
			for (int i = 0; i <  fftSize[0]; ++i) {
				Int3 coordinates(i, j, k);
				if (coordinates == Int3()) {
					BOOST_CHECK_GT(host_check_view[coordinates], 1.0);
					BOOST_CHECK_GT(device_check_view[coordinates], 1.0);
				} else {
					BOOST_CHECK_CLOSE(host_check_view[coordinates], 0.0, kFftEpsilon);
					BOOST_CHECK_CLOSE(device_check_view[coordinates], 0.0, kFftEpsilon);
				}
			}
		}
	}
}

template<int tDim>
void ForwardInverseTest(Vector<int, tDim> tiles, Vector<int, tDim> size){
	// Do forward and then inverse FFT and compare result with the input image - should be the same.
	auto view = checkerboard(1.0f, 0.0f, tiles, size);

	{// device
		FftCalculator<tDim,DeviceFftPolicy<Forward>> forward(size);
		auto image = forward.createSpaceDomainDeviceImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainDeviceImage();
		auto result = forward.createSpaceDomainDeviceImage();

		forward.calculate(image.view(), image_fft.view());


		BOLT_CHECK(cudaThreadSynchronize());

		FftCalculator<tDim,DeviceFftPolicy<Inverse>> inverse(result.size());
		inverse.calculateAsync(image_fft.view(), result.view());
		BOLT_CHECK(cudaThreadSynchronize());


		// FFT is without normalisation factor
		testViewsForIdentity(multiplyByFactor(1.0f / view.elementCount(), result.constView()), view, kFftEpsilon);

	}
	{// host
		FftCalculator<tDim,HostFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainHostImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainHostImage();
		auto result  = forward.createSpaceDomainHostImage();

		forward.calculate(image.view(), image_fft.view());
		FftCalculator<tDim,HostFftPolicy<Inverse>> inverse(result.size());
		inverse.calculateAndNormalize(image_fft.view(), result.view());

		testViewsForIdentity(result.constView(), view, kFftEpsilon);
	}
}



BOOST_AUTO_TEST_CASE(ForwardInverse) {
	ForwardInverseTest(Int1(2), Int1(32));
	ForwardInverseTest(Int2(2, 2), Int2(32, 16));
	ForwardInverseTest(Int3(2, 2, 2), Int3(32, 16, 16));
}


template <int tDim>
void checkSpectra(HostImage<DeviceComplexType, tDim> & fft_spectrum);

template<>
void checkSpectra(HostImage<DeviceComplexType, 1> & fft_spectrum){
	auto frequency_size = fft_spectrum.size();
	auto original_spectrum_view = fft_spectrum.constView();
	auto half_spectrum_view  = halfSpectrumView(original_spectrum_view);
	auto const_spectrum_view  = constSpectrumView(original_spectrum_view);

	BOOST_CHECK_EQUAL((topCorner(original_spectrum_view)), (Int1(0)));
	BOOST_CHECK_EQUAL((topCorner(original_spectrum_view) + half_spectrum_view.size()), (Int1(frequency_size[0])));

	auto top = topCorner(const_spectrum_view);
	BOOST_CHECK_EQUAL((top), (Int1(-frequency_size[0] + 1)));
	auto bot = top + const_spectrum_view.size();
	BOOST_CHECK_EQUAL((bot), (Int1(frequency_size[0])));

	for(int col = get(top, 0); col < get(bot, 0); col++){
		if(col >= 0) {
			BOOST_CHECK_EQUAL((const_spectrum_view[col].x), (original_spectrum_view[col].x));
			BOOST_CHECK_EQUAL((const_spectrum_view[col].y), (original_spectrum_view[col].y));
			BOOST_CHECK_EQUAL((half_spectrum_view[col].x), (original_spectrum_view[col].x));
			BOOST_CHECK_EQUAL((half_spectrum_view[col].y), (original_spectrum_view[col].y));
		}
		else {
			// Conjugate
			BOOST_CHECK_EQUAL((const_spectrum_view[col].x), (original_spectrum_view[std::abs(col)].x));
			BOOST_CHECK_EQUAL((const_spectrum_view[col].y), (-original_spectrum_view[std::abs(col)].y));
		}
	}
}

template<>
void checkSpectra(HostImage<DeviceComplexType, 2> & fft_spectrum){
	auto frequency_size = fft_spectrum.size();
	auto original_spectrum_view = fft_spectrum.constView();
	auto half_spectrum_view = halfSpectrumView(original_spectrum_view);
	auto const_spectrum_view = constSpectrumView(original_spectrum_view);

	BOOST_CHECK_EQUAL((topCorner(half_spectrum_view)), (Int2(0, -(frequency_size[1] - 1) / 2)));
	BOOST_CHECK_EQUAL(half_spectrum_view.size(), frequency_size);

	auto top = topCorner(const_spectrum_view);
	BOOST_CHECK_EQUAL((top), (Int2(-frequency_size[0] + 1, -frequency_size[1]/2)));
	auto bot = top + const_spectrum_view.size();
	BOOST_CHECK_EQUAL((bot), (Int2(frequency_size[0], frequency_size[1]/2 + 1)));

	for(int x = top[0]; x < bot[0]; x++){
		for(int y = top[1]; y < bot[1]; y++){
			Int2 xy (x, y);
			Int2 x_tr_y (x, (y < 0 ? frequency_size[1] + y : y));
			if(x >= 0) {
				BOOST_CHECK_EQUAL((const_spectrum_view[xy].x), (original_spectrum_view[x_tr_y].x));
				BOOST_CHECK_EQUAL((const_spectrum_view[xy].y), (original_spectrum_view[x_tr_y].y));
				BOOST_CHECK_EQUAL((half_spectrum_view[xy].x), (original_spectrum_view[x_tr_y].x));
				BOOST_CHECK_EQUAL((half_spectrum_view[xy].y), (original_spectrum_view[x_tr_y].y));
			}
			else {
				// Conjugate
				BOOST_CHECK_EQUAL((const_spectrum_view[xy].x), (half_spectrum_view[-xy].x));
				BOOST_CHECK_EQUAL((const_spectrum_view[xy].y), (-half_spectrum_view[-xy].y));
			}
		}
	}
}

template<>
void checkSpectra(HostImage<DeviceComplexType, 3> & fft_spectrum){
	auto frequency_size = fft_spectrum.size();
	auto original_spectrum_view = fft_spectrum.constView();
	auto half_spectrum_view  = halfSpectrumView(original_spectrum_view);
	auto const_spectrum_view  = constSpectrumView(original_spectrum_view);

	BOOST_CHECK_EQUAL((topCorner(half_spectrum_view)), (Int3(0, -(frequency_size[1] - 1) / 2, -(frequency_size[2] - 1) / 2)));
	BOOST_CHECK_EQUAL(half_spectrum_view.size(), frequency_size);

	auto top = topCorner(const_spectrum_view);
	BOOST_CHECK_EQUAL((top), (Int3(-frequency_size[0] + 1, -frequency_size[1]/2, -frequency_size[2]/2)));
	auto bot = top + const_spectrum_view.size();
	BOOST_CHECK_EQUAL((bot), (Int3(frequency_size[0], frequency_size[1]/2 + 1, frequency_size[2]/2 + 1)));

	for(int x = top[0]; x < bot[0]; x++){
		for(int y = top[1]; y < bot[1]; y++){
			int tr_y = (y < 0 ? frequency_size[1] + y : y);
			for(int z = top[2]; z < bot[2]; z++){
				int tr_z = (z < 0 ? frequency_size[2] + z : z);
				Int3 xyz (x, y, z);
				Int3 x_tr_yz (x, tr_y, tr_z);

				if(x >= 0) {
					BOOST_CHECK_EQUAL((const_spectrum_view[xyz].x), (original_spectrum_view[x_tr_yz].x));
					BOOST_CHECK_EQUAL((const_spectrum_view[xyz].y), (original_spectrum_view[x_tr_yz].y));
					BOOST_CHECK_EQUAL((half_spectrum_view[xyz].x), (original_spectrum_view[x_tr_yz].x));
					BOOST_CHECK_EQUAL((half_spectrum_view[xyz].y), (original_spectrum_view[x_tr_yz].y));
				}
				else {
					// Conjugate
					BOOST_CHECK_EQUAL((const_spectrum_view[xyz].x), (half_spectrum_view[-xyz].x));
					BOOST_CHECK_EQUAL((const_spectrum_view[xyz].y), (-half_spectrum_view[-xyz].y));
				}
			}
		}
	}
}



template <int tDim>
void SpectrumViewTest(Vector<int, tDim> tiles, Vector<int, tDim> size){
	static_assert(tDim >= 1 && tDim <= 3, "Supporting only 1D, 2D, 3D");
	auto frequency_size = getFftImageSize(size);

	auto view = checkerboard(1.0f, 0.0f, tiles, size);
	HostImage<DeviceComplexType, tDim> deviceResult(frequency_size);
	HostImage<DeviceComplexType, tDim> hostResult(frequency_size);

	{// device
		FftCalculator<tDim, DeviceFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainDeviceImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainDeviceImage();

		forward.calculate(image.view(), image_fft.view());

		copy(image_fft.constView(), deviceResult.view());
	}

	{// host
		FftCalculator<tDim,HostFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainHostImage();
		copy(view, image.view());

		forward.calculate(image.view(), hostResult.view());
	}

	checkSpectra(deviceResult);
	checkSpectra(hostResult);

}


BOOST_AUTO_TEST_CASE(SpectrumViews){
	SpectrumViewTest(Int1(8), Int1(32));
	SpectrumViewTest(Int2(8, 4), Int2(32, 16));
	SpectrumViewTest(Int3(8, 4, 4), Int3(32, 16, 16));
}

BOOST_AUTO_TEST_CASE(SpectrumViewsWrite){
	Int2 tiles (2, 4);
	Int2 size (16, 8);
	auto frequency_size = getFftImageSize(size);

	auto view = checkerboard(1.0f, 0.0f, tiles, size);

	{// device
		FftCalculator<2, DeviceFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainDeviceImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainDeviceImage();
		auto image_fft_copy = forward.createFrequencyDomainDeviceImage();

		forward.calculate(image.view(), image_fft.view());

		auto half_spectrum = halfSpectrumView(image_fft.constView());
		auto half_spectrum_copy = halfSpectrumView(image_fft_copy.view());

		copy(half_spectrum, half_spectrum_copy);
		testViewsForIdentity(amplitude(image_fft.constView()), amplitude(image_fft_copy.constView()), kFftEpsilon);
	}

	{// host
		FftCalculator<2, HostFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainHostImage();
		auto image_fft = forward.createFrequencyDomainHostImage();
		auto image_fft_copy = forward.createFrequencyDomainHostImage();

		copy(view, image.view());

		forward.calculate(image.view(), image_fft.view());
		auto half_spectrum = halfSpectrumView(image_fft.constView());
		auto half_spectrum_copy = halfSpectrumView(image_fft_copy.view());

		copy(half_spectrum, half_spectrum_copy);
		testViewsForIdentity(amplitude(image_fft.constView()), amplitude(image_fft_copy.constView()), kFftEpsilon);
	}
}

BOOST_AUTO_TEST_CASE(Small2DFft) {
	// Checking FFT values of a simple 2x2 checkerboard image
	Int2 size(2,2);

	auto input_view = checkerboard(-1.0f, 1.0f, Int2(1, 1), size);

	typedef cuFloatComplex Complex;
	Complex expected[4] = {
		{0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {4.0f, 0.0f}
	};

	HostImage<DeviceComplexType, 2> device_check(size);
	HostImage<HostComplexType, 2> host_check(size);

	{// device
		FftCalculator<2,DeviceFftPolicy<Forward>> fft_calculator(size);
		auto input_device = fft_calculator.createSpaceDomainDeviceImage();
		copy(input_view, input_device.view());

		auto image_fft = fft_calculator.createFrequencyDomainDeviceImage();
		fft_calculator.calculate(input_device.view(), image_fft.view());

	copy(image_fft.constView(), device_check.view());
	}
	{// host
		FftCalculator<2,HostFftPolicy<Forward>> fft_calculator(size);
		auto input_host = fft_calculator.createSpaceDomainHostImage();
		copy(input_view, input_host.view());

		auto image_fft = fft_calculator.createFrequencyDomainHostImage();
		fft_calculator.calculate(input_host.view(), image_fft.view());

		copy(image_fft.constView(), host_check.view());
	}

	auto host_check_view  = host_check.constView();
	auto device_check_view  = device_check.constView();

	for (int y = 0; y < size[1]; ++y) {
		for (int x = 0; x < size[0]; ++x) {
			BOOST_CHECK_SMALL(host_check_view[Int2(x, y)].x - expected[size[1] * y + x].x, (float)kFloatTestEpsilon);
			BOOST_CHECK_SMALL(host_check_view[Int2(x, y)].y - expected[size[1] * y + x].y, (float)kFloatTestEpsilon);

			BOOST_CHECK_SMALL(device_check_view[Int2(x, y)].x - expected[size[1] * y + x].x, (float)kFloatTestEpsilon);
			BOOST_CHECK_SMALL(device_check_view[Int2(x, y)].y - expected[size[1] * y + x].y, (float)kFloatTestEpsilon);
		}
	}

}

template<int tDim>
void phaseShiftTest(Vector<int, tDim> size){
	Vector<int, tDim> halfSize;
	Vector<float, tDim> sh;
	// NOTE(fidli): Vector1 operations return scalar, which we do not want here
	for(int i = 0; i < tDim; i++){
		halfSize[i] = size[0]/2;
		sh[i] = -halfSize[i];
	}
	auto view = checkerboard(1.0f, 0.0f, halfSize, size);
	auto check = checkerboard(0.0f, 1.0f, halfSize, size);

	{// device
		FftCalculator<tDim,DeviceFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainDeviceImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainDeviceImage();
		forward.calculateAndNormalize(image.view(), image_fft.view());


		auto image_fft_shifted = forward.createFrequencyDomainDeviceImage();
		copy(phaseShift(image_fft.view(), size, sh), image_fft_shifted.view());


		FftCalculator<tDim,DeviceFftPolicy<Inverse>> inverse(view.size());

		auto deviceSpaceShift = inverse.createSpaceDomainDeviceImage();
		auto hostSpaceShift = inverse.createSpaceDomainHostImage();
		inverse.calculate(image_fft_shifted.view(), deviceSpaceShift.view());
		copy(deviceSpaceShift.view(), hostSpaceShift.view());
		testViewsForIdentity(check, hostSpaceShift.view(), kFftEpsilon);
	}

	{// host
		FftCalculator<tDim,HostFftPolicy<Forward>> forward(view.size());
		auto image = forward.createSpaceDomainHostImage();
		copy(view, image.view());
		auto image_fft = forward.createFrequencyDomainHostImage();
		forward.calculateAndNormalize(image.view(), image_fft.view());

		auto image_fft_shifted = forward.createFrequencyDomainHostImage();
		copy(phaseShift(image_fft.view(), size, sh), image_fft_shifted.view());

		FftCalculator<tDim,HostFftPolicy<Inverse>> inverse(view.size());

		auto hostSpaceShift = inverse.createSpaceDomainHostImage();
		inverse.calculate(image_fft_shifted.view(), hostSpaceShift.view());
		testViewsForIdentity(check, hostSpaceShift.view(), kFftEpsilon);
	}

}

BOOST_AUTO_TEST_CASE(PhaseShifts){
	phaseShiftTest(Int1(16));
	phaseShiftTest(Int2(16, 16));
	phaseShiftTest(Int3(16, 16, 16));
}

template <int tDim>
void crossPowerSpectrumTestWithSines(Vector<int, tDim> size){
	Vector<float, tDim> amplitudes;
	Vector<float, tDim> frequencies;
	Vector<float, tDim> phaseOffset;
	for(int i = 0; i < tDim; i++){
		amplitudes[i] = 1;
		frequencies[i] = 1;
		phaseOffset[i] = kPi;
	}
	auto viewA = makeSinusImageView(size, amplitudes, frequencies, Vector<float, tDim>());
	auto viewB = makeSinusImageView(size, amplitudes, frequencies, phaseOffset);

	auto freqSize = getFftImageSize(size);

	DeviceImage<float, tDim> deviceSpaceImageA(size);
	DeviceImage<float, tDim> deviceSpaceImageB(size);
	copy(viewA, deviceSpaceImageA.view());
	copy(viewB, deviceSpaceImageB.view());

	FftCalculator<tDim, DeviceFftPolicy<Forward>> devCalc(size);
	FftCalculator<tDim, HostFftPolicy<Forward>> hostCalc(size);
	auto hostFreqA = hostCalc.createFrequencyDomainHostImage();
	auto hostFreqB = hostCalc.createFrequencyDomainHostImage();
	auto devFreqA = devCalc.createFrequencyDomainDeviceImage();
	auto devFreqB = devCalc.createFrequencyDomainDeviceImage();
	devCalc.calculateAndNormalize(deviceSpaceImageA.view(), devFreqA.view());
	devCalc.calculateAndNormalize(deviceSpaceImageB.view(), devFreqB.view());

	hostCalc.calculateAndNormalize(viewA, hostFreqA.view());
	hostCalc.calculateAndNormalize(viewB, hostFreqB.view());

	auto checkFreqA = devCalc.createFrequencyDomainHostImage();
	auto checkFreqB = devCalc.createFrequencyDomainHostImage();
	copy(devFreqA.view(), checkFreqA.view());
	copy(devFreqB.view(), checkFreqB.view());
	testViewsElementsForIdentity(checkFreqA.view(), cast<cufftComplex>(hostFreqA.view()), kFftEpsilon);
	testViewsElementsForIdentity(checkFreqB.view(), cast<cufftComplex>(hostFreqB.view()), kFftEpsilon);

	auto devConjugateB = devCalc.createFrequencyDomainDeviceImage();
	auto checkConjugateB = devCalc.createFrequencyDomainHostImage();
	copy(conjugate(devFreqB.view()), devConjugateB.view());
	copy(devConjugateB.view(), checkConjugateB.view());
	testViewsElementsForIdentity(checkConjugateB.view(), cast<cufftComplex>(conjugate(hostFreqB.view())), kFftEpsilon);

	auto devHadamard = devCalc.createFrequencyDomainDeviceImage();
	auto checkHadamard = devCalc.createFrequencyDomainHostImage();
	copy(hadamard(devFreqA.view(), devConjugateB.view()), devHadamard.view());
	copy(devHadamard.view(), checkHadamard.view());
	testViewsElementsForIdentity(checkHadamard.view(), cast<cufftComplex>(hadamard(hostFreqA.view(), conjugate(hostFreqB.view()))), kFftEpsilon);

	auto devNormalize = devCalc.createFrequencyDomainDeviceImage();
	auto checkNormalize = devCalc.createFrequencyDomainHostImage();
	copy(normalize(devHadamard.view()), devNormalize.view());
	copy(devNormalize.view(), checkNormalize.view());
	testViewsElementsForIdentity(checkNormalize.view(), cast<cufftComplex>(normalize(hadamard(hostFreqA.view(), conjugate(hostFreqB.view())))), kFftEpsilon);

	auto deviceCrossPower = devCalc.createFrequencyDomainDeviceImage();
	copy(crossPowerSpectrum(devFreqA.view(), devFreqB.view()), deviceCrossPower.view());
	auto deviceCrossPowerCheck = devCalc.createFrequencyDomainHostImage();
	copy(deviceCrossPower.view(), deviceCrossPowerCheck.view());
	auto hostCrossPower = crossPowerSpectrum(hostFreqA.view(), hostFreqB.view());

	testViewsElementsForIdentity(cast<cufftComplex>(hostCrossPower), deviceCrossPowerCheck.view(), kFftEpsilon);
}

BOOST_AUTO_TEST_CASE(CrossPowerSpectrums){
	crossPowerSpectrumTestWithSines(Int1(16));
	crossPowerSpectrumTestWithSines(Int2(16, 16));
	crossPowerSpectrumTestWithSines(Int3(16, 16, 16));
}

template <int tDim>
void phaseCorrelationTestWithQuads(Vector<int, tDim> size){
	Vector<int, tDim> shift;
	for(int i = 0; i < tDim; i++){
		set(shift, i, get(size, i)/2);
	}

	// generating shifted quad
	HostImage<float, tDim> imgA(size);
	HostImage<float, tDim> imgB(size);
	for(int i = 0; i < product(size); i++){
		{
			auto index = getIndexFromLinearAccessIndex(imgA.view(), i);
			bool white = true;
			for(int j = 0; white && j < tDim; j++){
				white &= get(index, j) < (get(shift, j) - 1) && get(index, j) > 1;
			}
			if(white){
				linearAccess(imgA.view(), i) = 1.0f;
			}else{
				linearAccess(imgA.view(), i) = 0.0f;
			}
		}
		{
		auto index = getIndexFromLinearAccessIndex(imgB.view(), i);
		bool white = true;
		for(int j = 0; white && j < tDim; j++){
			white &= get(index, j) > (get(shift, j) + 1) && get(index, j) < (get(size, j) - 1);
		}
		if(white){
			linearAccess(imgB.view(), i) = 1.0f;
		}else{
			linearAccess(imgB.view(), i) = 0.0f;
		}
		}
	}
	auto viewA = imgA.constView();
	auto viewB = imgB.constView();

	auto freqSize = getFftImageSize(size);

	DeviceImage<float, tDim> spaceA(size);
	DeviceImage<float, tDim> spaceB(size);
	copy(viewA, spaceA.view());
	copy(viewB, spaceB.view());

	auto deviceCorrelation = phaseCorrelation(spaceA.view(), spaceB.view());
	auto hostCorrelation = phaseCorrelation(viewA, viewB);

	HostImage<float, tDim> deviceCorrelationHost(size);
	copy(deviceCorrelation.view(), deviceCorrelationHost.view());
	testViewsElementsForIdentity(hostCorrelation.view(), deviceCorrelationHost.view(), kFftEpsilon);

	// find the peak, it should be somewhere in the half of the image
	// ignore borders of 1px
	// @Todo reduction (max) with coordinates
	Vector<int, tDim> maxCoords;
	float max = -std::numeric_limits<float>::infinity();
	for(int i = 0; i < product(size); i++){
		auto coords = getIndexFromLinearAccessIndex(hostCorrelation.view(), i);
		bool isBorder = false;
		for(int j = 0; j < tDim; j++){
			isBorder |= get(coords, j) == 0;
		}
		if(!isBorder){
			if(hostCorrelation.view()[coords] > max){
				max = hostCorrelation.view()[coords];
				maxCoords = coords;
			}
		}
	}

	// tolerance of 2 pixels
	for(int i = 0; i < tDim; i++){
		BOOST_CHECK((get(maxCoords, i) >= (get(shift, i) - 2)) && ((get(maxCoords, i) <= get(shift, i) + 2)));
	}
}

// https:// en.wikipedia.org/wiki/Phase_correlation
BOOST_AUTO_TEST_CASE(PhaseCorrelations){
	phaseCorrelationTestWithQuads(Int1(16));
	phaseCorrelationTestWithQuads(Int2(16, 16));
	phaseCorrelationTestWithQuads(Int3(16, 16, 16));

}

template<int tStackDim, int tDim>
void StackTestWithConstants(Vector<int, tDim> size){
	static_assert(tStackDim >= 0, "this is a stack test, stack it!");
	auto sub_size = subSize<Stack<tStackDim>>(size);

	FftCalculator<tDim-1, DeviceFftPolicy<Forward, Stack<tStackDim>>> devStackForward(size);
	FftCalculator<tDim-1, HostFftPolicy<Forward, Stack<tStackDim>>> hostStackForward(size);
	FftCalculator<tDim-1, HostFftPolicy<Forward>> hostForward(sub_size);
	FftCalculator<tDim-1, DeviceFftPolicy<Forward>> devForward(sub_size);

	FftCalculator<tDim-1, DeviceFftPolicy<Inverse, Stack<tStackDim>>> devStackInverse(size);
	FftCalculator<tDim-1, HostFftPolicy<Inverse, Stack<tStackDim>>> hostStackInverse(size);
	FftCalculator<tDim-1, HostFftPolicy<Inverse>> hostInverse(sub_size);
	FftCalculator<tDim-1, DeviceFftPolicy<Inverse>> devInverse(sub_size);


	auto hostFreqs = std::unique_ptr<std::unique_ptr<HostImage<HostComplexType, tDim-1>>[]>(new std::unique_ptr<HostImage<HostComplexType, tDim-1>>[get(size, tStackDim)]);
	auto checkFreqs = std::unique_ptr<std::unique_ptr<HostImage<cufftComplex, tDim-1>>[]>(new std::unique_ptr<HostImage<cufftComplex, tDim-1>>[get(size, tStackDim)]);

	auto hostSpaces = std::unique_ptr<std::unique_ptr<HostImage<float, tDim-1>>[]>(new std::unique_ptr<HostImage<float, tDim-1>>[get(size, tStackDim)]);
	auto checkSpaces = std::unique_ptr<std::unique_ptr<HostImage<float, tDim-1>>[]>(new std::unique_ptr<HostImage<float, tDim-1>>[get(size, tStackDim)]);


	auto hostStackSpace = hostStackForward.createSpaceDomainHostImage();
	auto devStackSpace = devStackForward.createSpaceDomainDeviceImage();

	auto hostStackResult = hostStackInverse.createSpaceDomainHostImage();
	auto devStackResult = devStackInverse.createSpaceDomainDeviceImage();
	auto checkStackResult = devStackInverse.createSpaceDomainHostImage();

	int inputs = get(size, tStackDim);

	// forward part
	for(int inputIndex = 0; inputIndex < inputs; inputIndex++){
		auto inputData = makeConstantImageView(static_cast<float>(inputIndex), sub_size);
		// start copy to stack
		for(int inputDataPixelIndex = 0; inputDataPixelIndex < product(sub_size); inputDataPixelIndex++){
			auto sourceCoords = getIndexFromLinearAccessIndex(inputData, inputDataPixelIndex);
			Vector<int, tDim> spaceCoords;
			int wi = 0;
			for(int i = 0; i < tDim; i++){
				if(i == tStackDim){
					set(spaceCoords, i, inputIndex);
				}else{
					set(spaceCoords, i, get(sourceCoords, wi));
					wi++;
				}
			}
			hostStackSpace.view()[spaceCoords] = inputData[sourceCoords];
		}
		// end copy to stack

		// calculate single
		hostFreqs.get()[inputIndex].reset(new HostImage<HostComplexType, tDim-1>(getFftImageSize(sub_size)));
		hostForward.calculateAndNormalize(inputData, hostFreqs.get()[inputIndex].get()->view());

		auto devInput = devForward.createSpaceDomainDeviceImage();
		copy(inputData, devInput.view());
		auto devOutput = devForward.createFrequencyDomainDeviceImage();
		devForward.calculateAndNormalize(devInput.view(), devOutput.view());

		checkFreqs.get()[inputIndex].reset(new HostImage<cufftComplex, tDim-1>(getFftImageSize(sub_size)));
		copy(devOutput.view(), checkFreqs.get()[inputIndex].get()->view());

	}

	copy(hostStackSpace.view(), devStackSpace.view());

	auto hostStackFreq = hostStackForward.createFrequencyDomainHostImage();
	auto devStackFreq = devStackForward.createFrequencyDomainDeviceImage();

	hostStackForward.calculateAndNormalize(hostStackSpace.view(), hostStackFreq.view());
	devStackForward.calculateAndNormalize(devStackSpace.view(), devStackFreq.view());

	auto checkStackFreq = devStackForward.createFrequencyDomainHostImage();
	copy(devStackFreq.view(), checkStackFreq.view());

	testViewsElementsForIdentity(checkStackFreq.view(), cast<cufftComplex>(hostStackFreq.view()), kFftEpsilon);

	// CHECK frequency stack to non stack
	for(int inputIndex = 0; inputIndex < inputs; inputIndex++){
		testViewsElementsForIdentity(cast<cufftComplex>(hostFreqs.get()[inputIndex].get()->view()), checkFreqs.get()[inputIndex].get()->view(), kFftEpsilon);
		testViewsElementsForIdentity(hostFreqs.get()[inputIndex].get()->view(), slice<tStackDim>(hostStackFreq.view(), inputIndex), 0);
		// NOTE(fidli): it seems that batched fft vs non batched fft makes difference, but neglectable
		testViewsElementsForIdentity(checkFreqs.get()[inputIndex].get()->view(), slice<tStackDim>(checkStackFreq.view(), inputIndex), 0.000000001);
	}

	// inverse part
	hostStackInverse.calculate(hostStackFreq.view(), hostStackResult.view());
	devStackInverse.calculate(devStackFreq.view(), devStackResult.view());
	copy(devStackResult.view(), checkStackResult.view());

	testViewsForIdentity(checkStackResult.view(), hostStackResult.view(), kFftEpsilon);
	testViewsForIdentity(hostStackResult.view(), hostStackSpace.view(), kFftEpsilon);

	for(int inputIndex = 0; inputIndex < inputs; inputIndex++){
		auto inputData = makeConstantImageView(static_cast<float>(inputIndex), sub_size);

		auto hostR = hostInverse.createSpaceDomainHostImage();
		hostInverse.calculate(hostFreqs.get()[inputIndex].get()->view(), hostR.view());

		auto devR = devInverse.createSpaceDomainDeviceImage();
		auto checkR = devInverse.createSpaceDomainHostImage();
		auto devIn = devInverse.createFrequencyDomainDeviceImage();
		copy(checkFreqs.get()[inputIndex].get()->view(), devIn.view());
		devInverse.calculate(devIn.view(), devR.view());
		copy(devR.view(), checkR.view());

		testViewsForIdentity(checkR.view(), hostR.view(), kFftEpsilon);

		// NOTE(fidli): it seems that batched fft vs non batched fft makes difference, but neglectable
		testViewsForIdentity(checkR.view(), slice<tStackDim>(checkStackResult.view(), inputIndex), 0.0000000001);
		testViewsForIdentity(hostR.view(), slice<tStackDim>(hostStackResult.view(), inputIndex), 0);

		testViewsForIdentity(hostR.view(), inputData, kFftEpsilon);
	}

}

BOOST_AUTO_TEST_CASE(Stacks){
	StackTestWithConstants<0>(Int2(1, 16));
	StackTestWithConstants<1>(Int2(16, 1));

	StackTestWithConstants<0>(Int2(11, 34));
	StackTestWithConstants<1>(Int2(34, 11));

	StackTestWithConstants<0>(Int3(1, 16, 24));
	StackTestWithConstants<1>(Int3(16, 1, 24));
	StackTestWithConstants<2>(Int3(16, 24, 1));

	StackTestWithConstants<0>(Int3(11, 33, 7));
	StackTestWithConstants<1>(Int3(33, 11, 7));
	StackTestWithConstants<2>(Int3(33, 7, 11));

	// TODO(fidli): Also do tests with double stacks, e.g FftCalculator<1, Device/HostFftPolicy<Forward/Inverse, Stack<0,1>> etc...
}

}  // namespace bolt
