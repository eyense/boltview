#include "edge_detection.h"
#include <boltview/for_each.h>
#include <boltview/transform.h>
#include <eyen/core/math.h>
#include "ecip/host_image.h"
#include "ecip/copy.h"
#include <iostream>
#include "ecip/host_image_view.h"
#include "ecip/convolution_kernels.h"
#include "ecip/convolution.h"
#include "ecip/procedural_views.h"
#include "ecip/functors.h"
#include <math.h>
#include "thresholding.h"
#include "process_qimage.h"

using namespace eyen;
using namespace ecip;

float mat_h[] = { 1, 2, 1,
                 0, 0, 0,
                 -1, -2, -1};


float mat_v[] = { -1, 0, 1,
                 -2, 0, 2,
                 -1, 0, 1};


const int kThresholdingLimit = 127;


struct SqrtFunctor {
    float operator ()(float value) {
        return sqrt(value);
     }
};


template<typename TType>
struct PowFunctor {
    explicit PowFunctor(TType value) :
        _value(value)
    {}

    template<typename T>
    void operator()(T &in_value) const {
        float product = 1;
        for (int i = 0; i < _value; i++) {
            product *= in_value;
        }
        in_value = product;
    }

    TType _value;
};


template<typename TView>
void ConvolutionView(TView & input_view, TView &output_view, float  mat[]) {
    Convolution(input_view, output_view, DynamicUnifiedKernel<float, 2>(Int2(3, 3), Int2(1, 1), mat));
    ForEach(output_view, PowFunctor<int>(2));
}


void EdgeDetection (QImage & input_image, QImage & output_image) {
    auto input_image_view = ToView(input_image);
    auto output_image_view = ToView(output_image);
    const Int2 size_of_image(input_image.size().width(), input_image.size().height());
    HostImage<float, 2> output_h(size_of_image);
    HostImage<float, 2> output_v(size_of_image);
    auto input_view = ToFloatView(input_image_view, size_of_image);
    auto output_view_h = output_h.View();
    auto output_view_v = output_v.View();
    HostImage<float, 2> output(size_of_image);
    auto output_view = output.View();
    ConvolutionView<HostImageView<float, 2>>(input_view, output_view_h, mat_h);
    ConvolutionView<HostImageView<float, 2>>(input_view, output_view_v, mat_v);
    auto sum_view = Add(output_view_v, output_view_h);
    Transform(sum_view, output_view, SqrtFunctor{});
    Copy(output_view, output_image_view);
    Thresholding(output_image, kThresholdingLimit);
}



