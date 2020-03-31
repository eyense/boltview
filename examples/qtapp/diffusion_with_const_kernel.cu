#include <math.h>
#include "ecip/host_image.h"
#include "ecip/host_image_view.h"
#include "ecip/convolution_kernels.h"
#include "ecip/convolution.h"
#include "ecip/image_io.h"
#include <eyen/core/math.h>
#include <iostream>
#include <boltview/transform.h>
#include "diffusion_with_const_kernel.h"
#include "process_qimage.h"
using namespace eyen;
using namespace ecip;



float mat[] = {0, 1, 0,
               1, -4, 1,
               0, 1, 0};


template<typename TView>
void StartDiffusionWithConstKernel (TView & input_view, TView & output_image, int iterations_number) {
    for (int i = 0; i < iterations_number; i++) {
        Convolution(input_view, output_image, DynamicUnifiedKernel<float, 2>(Int2(3, 3), Int2(1, 1), mat));
        std::swap(input_view, output_image);
    }
}


void DiffusionWithConstKernel(QImage & input_image, QImage & output_image, int iterations_number) {
    auto input_image_bits_iterator = input_image.bits();
    const Int2 size_of_image(input_image.size().width(), input_image.size().height());
    auto output_image_bits_iterator = output_image.bits();
    auto in = MakeHostImageView(input_image_bits_iterator, size_of_image);
    auto out = MakeHostImageView(output_image_bits_iterator, size_of_image);
    HostImage<float, 2> input(size_of_image);
    HostImage<float, 2> output(size_of_image);
    auto input_view = input.View();
    auto output_view = output.View();
    Copy(in, input_view);
    StartDiffusionWithConstKernel(input_view, output_view, iterations_number);
    Transform(input_view, out, ClampFunctor{});
}

