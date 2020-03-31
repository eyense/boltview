#include "anisotropic.h"
#include <math.h>
#include "ecip/host_image.h"
#include "ecip/host_image_view.h"
#include "ecip/convolution_kernels.h"
#include "ecip/convolution.h"
#include "ecip/image_io.h"
#include <eyen/core/math.h>
#include <boltview/transform.h>
#include "process_qimage.h"
#include <iostream>


using namespace eyen;
using namespace ecip;


void AnisotropicDiffusion (QImage & input_image, QImage & output_image, int iterations_number, int coefficient, float speed) {
    auto in = ToView(input_image);
    auto out = ToView(output_image);
    const Int2 size_of_input_image(input_image.size().width(), input_image.size().height());
    HostImage<float, 2> input(size_of_input_image);
    HostImage<float, 2> output(size_of_input_image);
    auto input_view = input.View();
    auto output_view = output.View();
    Copy(in, input_view);
    float mask[] = {1, 1, 1,
                    1, 1, 1,
                    1, 1, 1};
    for (int i = 0; i < iterations_number; i++) {
        StartAnisotropicDiffusion(input_view, output_view,  DynamicHostKernel<float, 2>(Int2(3, 3), Int2(1, 1), mask), coefficient, speed);
        std::swap(input_view, output_view);
    }
    Transform(input_view, out, ClampFunctor{});
}


BOLT_HD_WARNING_DISABLE
template<typename type,  typename TCallable, typename TLocator>
float PixelDiffusion(int coefficient, type from, type to, TCallable callable, TLocator locator)
{
    float result = 0;
    type index;
    for(index[1] = from[1]; index[1] < to[1]; ++index[1]) {
       for (index[0] = from[0]; index[0] < to[0]; ++index[0]) {
           float med =  callable(index) - locator.Get();
           float w = exp(-sqr(med / coefficient));
           result += w * med;
       }
    }
    return result;
}

///Functor that returns result pixel value
template<typename TConvolutionKernel>
struct DifFunctor{
      TConvolutionKernel _kernel;
      int _coefficient;
      float _speed;


      explicit DifFunctor(TConvolutionKernel kernel, int coefficient, float speed):
                 _kernel(kernel), _coefficient(coefficient), _speed(speed)
                 {}


      template<typename TLocator>
      float operator ()(TLocator locator) {
          float sum = 0;
          sum = PixelDiffusion(
                       _coefficient,
                       KernelStart(_kernel),
                       KernelEnd(_kernel),
                       [&](Vector<int, 2> index){ return locator[index] * _kernel[index]; },
                       locator);
          float result  = _speed * sum + locator.Get();
          return result;
       }
};


template<typename TConvolutionKernel, typename TView>
void StartAnisotropicDiffusion (TView & input_view, TView & output_view, const TConvolutionKernel  & kernel, int coefficient, float speed) {
    TransformLocator(input_view, output_view, DifFunctor<
                     typename KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::Type>
                             (KernelView<TConvolutionKernel, TConvolutionKernel::kIsDynamicallyAllocated>::Get(kernel),
                              coefficient, speed));
}


