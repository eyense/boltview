
#include <string>
#include <iostream>

#include <boltview/host_image.h>
#include <boltview/copy.h>
#include <boltview/reduce.h>
#include <boltview/create_view.h>
#include <boltview/convolution.h>
#include <boltview/subview.h>
#include <boltview/convolution_kernels.h>
#include <boltview/image_io.h>
#include <boltview/loop_utils.h>
#include <boltview/fft/fft_calculator.h>
#include <boltview/for_each.h>
#include <boltview/math/vector.h>
#include "io.h"


using namespace bolt;

struct KernelData {
	int width = 0;
	int height = 0;

	std::vector<float> weights;
};

KernelData loadKernel(boost::filesystem::path file) {
	std::ifstream f;
	f.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
	f.open(file.string());

	KernelData d;
	f >> d.width >> d.height;

	d.weights.resize(d.width * d.height);

	for (auto &v : d.weights) {
		f >> v;
	}
	return d;
}

template<typename TKernel>
BOLT_DECL_HYBRID
auto flipConvolutionKernel(const TKernel &k) {
	auto kernel_view = mirror(makeHostImageConstView(k.pointer(), k.size()), Bool2(true, true));
	HostImage<float, 2> tmp_kernel(k.size());
	copy(kernel_view, view(tmp_kernel));

	return TKernel(
			k.size(),
			k.size() - k.center() - Int2(1, 1),
			tmp_kernel.pointer());
}

template<typename TKernel, typename TSize>
BOLT_DECL_HYBRID
auto padConvolutionKernel(TKernel &k, TSize size) {
	auto kernel_view = mirror(makeHostImageConstView(k.pointer(), k.size()), Bool2(true, true));
	auto new_center = k.size() - k .center();
	return paddedView(kernel_view, size, -new_center + Int2(1, 1), 0);
}


struct WienerFtor {

	template<typename TValue>
	BOLT_DECL_HYBRID
	auto operator()(const TValue &val) const {
		return conjugate(val) / (magSquared(val) + noise_factor);
	}

	float noise_factor = 0.0f;
};

template<typename TPsf>
auto wiener(TPsf psf, float noise_factor) {
	return UnaryOperatorImageView<TPsf, WienerFtor>(psf, WienerFtor{noise_factor});
}

template<typename TView, typename TPsf>
void wienerDeconvolution(TView im, TPsf psf) {
	DeviceImage<float, 2> d_image(im.size());
	DeviceImage<float, 2> d_psf(im.size());
	copy(im, view(d_image));
	copy(psf, view(d_psf));

	FftCalculator<2, DeviceFftPolicy<Forward>> forward(im.size());
	auto image_freq = forward.createFrequencyDomainDeviceImage();
	auto psf_freq = forward.createFrequencyDomainDeviceImage();

	forward.calculate(view(d_psf), view(psf_freq));
	forward.calculate(view(d_image), view(image_freq));

	copy(view(image_freq) * wiener(view(psf_freq), 0.002f), view(image_freq));

	FftCalculator<2, DeviceFftPolicy<Inverse>> inverse(d_image.size());
	inverse.calculateAndNormalize(view(image_freq), view(d_image));

	copy(constView(d_image), im);
}

template<typename TView, typename TPsf>
void richardsonLucyDeconvolution(TView im, const TPsf &host_psf) {
	DeviceImage<float, 2> d_image(im.size());
	DeviceImage<float, 2> estimate(im.size());
	DeviceImage<float, 2> tmp(im.size());
	DeviceImage<float, 2> tmp2(im.size());
	copy(im, view(d_image));
	copy(constView(d_image), view(estimate));

	DynamicDeviceKernel<float, 2> psf(host_psf);
	DynamicDeviceKernel<float, 2> psf_hat(flipConvolutionKernel(host_psf));

	const int kIterationCount = 100;
	int s = 0;
	DeviceImage<float, 3> iterations(im.size()[0], im.size()[1], kIterationCount/10);
	for (int i = 0; i < kIterationCount; ++i) {
		convolution(constView(estimate), view(tmp), psf);
		copy(constView(d_image) / constView(tmp), view(tmp));
		convolution(constView(tmp), view(tmp2), psf_hat);
		copy(constView(tmp2) * constView(estimate), view(estimate));

		if (0 == (i + 1) % 10) {
			copy(constView(estimate), slice<2>(view(iterations), s));
			++s;
		}
		std::cout << "Iteration :" << i << "\n";
	}
	dump(constView(iterations), "_RL_deconv");

	copy(constView(estimate), im);
}

int main(int argc, char** argv) {
	try {
		// auto data = loadKernel("blur.txt");

#if 0
		std::vector<DynamicHostKernel<float, 2>> kernels;
		{
			HostImage<float, 2> kernel(Int2(25, 25));
			kernel.clear();

			view(kernel)[div(kernel.size(), 2)] = 255;
			{
				DynamicHostKernel<float, 2> k(
						kernel.size(),
						div(kernel.size(), 2),
						kernel.pointer());
				auto p = padConvolutionKernel(k, Int2(60,60));
				dump(p, "k1");
				kernels.push_back(std::move(k));
			}

			view(kernel)[Int2()] = 128;
			view(kernel)[Int2(24, 24)] = 200;
			{
				DynamicHostKernel<float, 2> k(
						kernel.size(),
						div(kernel.size(), 2),
						kernel.pointer());
				auto p = padConvolutionKernel(k, Int2(60,60));
				dump(p, "k2");
				kernels.push_back(std::move(k));
			}

			kernel.clear();
		}

		{
			HostImage<float, 2> kernel(Int2(25, 25));
			kernel.clear();

			view(kernel)[div(kernel.size(), 2)] = 255;
			{
				DynamicHostKernel<float, 2> k(
						kernel.size(),
						Int2(24, 24),
						kernel.pointer());
				auto p = padConvolutionKernel(k, Int2(60,60));
				dump(p, "ks1");
				kernels.push_back(std::move(k));
			}

			view(kernel)[Int2()] = 128;
			view(kernel)[Int2(24, 24)] = 200;
			{
				DynamicHostKernel<float, 2> k(
					kernel.size(),
					Int2(24, 24),
					kernel.pointer());
				auto p = padConvolutionKernel(k, Int2(60,60));
				dump(p, "ks2");
				kernels.push_back(std::move(k));
			}

			kernel.clear();

		}

		HostImage<float, 2> input_image(Int2(800, 800));
		HostImage<float, 2> output_image(input_image.size());
		HostImage<float, 2> padded_kernel(input_image.size());
		auto checkers = makeCheckerBoardImageView(255, 0, Int2(128, 128), div(input_image.size(), 2));
		copy(paddedView(checkers, input_image.size(), div(input_image.size(), 4), 0), view(input_image));

		DeviceImage<float, 2> padded_kernel_device(input_image.size());
		DeviceImage<float, 2> image_device(input_image.size());
		FftCalculator<2, DeviceFftPolicy<Forward>> forward(image_device.size());
		auto image_freq = forward.createFrequencyDomainDeviceImage();
		auto kernel_freq = forward.createFrequencyDomainDeviceImage();

		copy(constView(input_image), view(image_device));
		forward.calculate(view(image_device), view(image_freq));


		FftCalculator<2, DeviceFftPolicy<Inverse>> inverse(image_device.size());
		int i = 1;
		for (auto &k: kernels) {
			std::string prefix = std::to_string(i) + "_spatial";
			convolution(constView(input_image), view(output_image), k);

			dump(constView(output_image), prefix);

			//*****************************
			copy(padConvolutionKernel(k, input_image.size()), view(padded_kernel));
			copy(view(padded_kernel), view(padded_kernel_device));
			forward.calculate(view(padded_kernel_device), view(kernel_freq));

			copy(view(kernel_freq) * view(image_freq), view(kernel_freq));
			inverse.calculateAndNormalize(view(kernel_freq), view(image_device));
			copy(constView(image_device), view(output_image));
			prefix = std::to_string(i) + "_freq";
			dump(constView(output_image), prefix);
			++i;
		}

#endif
		HostImage<float, 2> kernel(Int2(100, 100));
		HostImage<uint32_t, 2> kernelc(kernel.size());
		load(view(kernelc), "../kernel2");

		float s = sum(view(kernelc), 0.0f);
		copy((1.0f / s) * view(kernelc), view(kernel));

		// kernel.clear();
		// // view(kernel)[Int2(24,24)] = 1.0f;
		// view(kernel)[Int2()] = 1.0f;

		DynamicHostKernel<float, 2> k(
				kernel.size(),
				// Int2(),
				div(kernel.size(), 2),
				kernel.pointer());

		std::string prefix = "../../data/bee";

		auto imageVar = loadImage("blurred_bee.png");

		auto image = boost::get<HostImage<uint8_t, 2>>(std::move(imageVar));

		std::cout << image.size() << "\n";
		HostImage<float, 2> blurred_image(image.size());
		copy(constView(image), view(blurred_image));

		if (true)
		{
			HostImage<float, 2> padded_psf(blurred_image.size());
			copy(padConvolutionKernel(k, blurred_image.size()), view(padded_psf));

			wienerDeconvolution(view(blurred_image), constView(padded_psf));

			copy(cast<uint8_t>(clamp(round(subview(constView(blurred_image), Int2(), image.size())), 0, 255)), view(image));
			saveImage("deconvolved_bee.jpg", view(image));
		}

		if(false)
		{

			richardsonLucyDeconvolution(view(blurred_image), k);
			copy(cast<uint8_t>(clamp(round(subview(constView(blurred_image), Int2(), image.size())), 0, 255)), view(image));
			saveImage("rl_deconvolved_bee.jpg", view(image));
		}




#if 0
			// HostImage<uint8_t, 2> input_image(Int2(800, 800));
			HostImage<uint8_t, 2> input_image(Int2(4282, 2848));
			HostImage<float, 2> output_image(input_image.size());

			load(view(input_image), prefix);

			// auto checkers = makeCheckerBoardImageView(255, 0, Int2(128, 128), div(input_image.size(), 2));
			// copy(paddedView(checkers, input_image.size(), div(input_image.size(), 4), 0), view(input_image));

			convolution(constView(input_image), view(output_image), k);
			// copy(constView(input_image), view(output_image));

			// dump(constView(input_image), "input");
			// dump(constView(output_image), "output");
			copy(cast<uint8_t>(round(constView(output_image))), view(input_image));
			saveImage("blurred_bee.jpg", view(input_image));
		{
			HostImage<float, 2> padded_kernel(input_image.size());
			// copy(paddedView(constView(kernel), input_image.size(), div(kernel.size(), 2), 0.0f), view(padded_kernel));
			copy(padConvolutionKernel(k, input_image.size()), view(padded_kernel));

			DeviceImage<float, 2> padded_kernel_device(input_image.size());
			DeviceImage<float, 2> image_device(input_image.size());
			copy(constView(padded_kernel), view(padded_kernel_device));
			copy(constView(output_image), view(image_device));

			FftCalculator<2, DeviceFftPolicy<Forward>> forward(image_device.size());


			auto image_freq = forward.createFrequencyDomainDeviceImage();
			auto kernel_freq = forward.createFrequencyDomainDeviceImage();
			forward.calculate(view(padded_kernel_device), view(kernel_freq));
			forward.calculate(view(image_device), view(image_freq));

			copy(view(kernel_freq) * view(image_freq), view(image_freq));

			FftCalculator<2, DeviceFftPolicy<Inverse>> inverse(image_device.size());
			inverse.calculateAndNormalize(view(image_freq), view(image_device));

			dump(constView(image_device), "output2");


			copy(view(image_freq) * wiener(view(kernel_freq), 0.0f), view(image_freq));
			inverse.calculateAndNormalize(view(image_freq), view(image_device));
			dump(constView(image_device), "output3");
			//div(conjugate(kernel_freq), magSquared(kernel_freq) + constView


		}
		{


		}
#endif
		///dump(view(kernel), "kernel_norm");

		// HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(4282, 2848));
		// // HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(30000, 20000));
		// DeviceImage<Vector<uint8_t, 3>, 2> device_image(output_image.size());
                //
		// forEachPosition(view(device_image), Mandelbrot{device_image.size()});
                //
		// copy(constView(device_image), view(output_image));
		// save("output.jpg", constView(output_image));
	} catch (std::exception &e) {
		std::cout << "boost::diagnostic_information(e):\n" << boost::diagnostic_information(e) << std::endl;
		return 1;
	}

	return 0;

}
