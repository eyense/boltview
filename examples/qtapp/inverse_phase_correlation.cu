#include <algorithm>
/*#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <complex>*/
#include <cufft.h>
#include "inverse_phase_correlation.h"
#include "process_qimage.h"
#include "ecip/fft_utils_experimental.tcc"
#include "ecip/fft_utils_math.tcc"
#include "process_qimage.h"
#include "ecip/image_io.h"
#include "ecip/host_image.h"
#include "ecip/host_image_view.h"
#include "ecip/convolution_kernels.h"
#include "ecip/convolution.h"
#include "ecip/subview.h"
#include "ecip/transform.h"
namespace eyen{
namespace ecip{
namespace experimental{


struct PhaseFunctor{
    Int2 size_res;
    Int2 max_coords;
    PhaseFunctor(Int2 size_res, Int2 max_coords): size_res(size_res), max_coords(max_coords){}


    template<typename TLocator>
    float operator ()(TLocator locator) {
        return locator[(-(max_coords))]; }
};
void InversePhaseCorrelation(QImage &im, QImage &correlation_res, QImage &out_im, int x, int y, int size_w, int size_h) {
    Int2 size_res(size_w, size_h);
    Int2 topleft(x, y);
    Int2 size_input_image(im.size().width(), im.size().height());
    auto in = ToView(im);
    auto out = ToView(out_im);
    auto cor_res = ToView(correlation_res);
    HostImage<float, 2> input(size_input_image);
    HostImage<float, 2> output(size_res);
    HostImage<float, 2> res_cor(size_input_image);
    auto in_view = input.View();
    auto out_view = output.View();
    auto cor_view = res_cor.View();
    Copy(in, in_view);
    Copy(cor_res, cor_view);
    FftCalculator<2, HostFftPolicy<Forward>> forward(size_input_image);
    FftCalculator<2, HostFftPolicy<Inverse>> inverse(size_res);
    auto frequency_in = forward.CreateFrequencyDomainHostImage();
    forward.CalculateAndNormalize(in_view, frequency_in.View());
    auto sub_view_in = Subview(frequency_in.View(), topleft, size_res);
    auto cor_res_view = Subview(cor_view, topleft, size_res);
    Int2 max_coords;
    float max = -std::numeric_limits<float>::infinity();
    for(int i = 0; i < Product(size_res); i++){
        auto coords = GetIndexFromLinearAccessIndex(cor_res_view, i);
        bool is_border = false;
        for(int j = 0; j < 2; j++){
            is_border |= Get(coords, j) == 0;
        }
        if(!is_border){
            if(cor_res_view[coords] > max){
               max = cor_res_view[coords];
               max_coords = coords;
             }
         }
    }
    auto out_subview = Subview(in_view, topleft - max_coords, size_res);

    //TODO Finding the proper image shift from phase correlation peak location


    /*auto frequency_out = inverse.CreateFrequencyDomainHostImage();
    inverse.CalculateAndNormalize(out_subview, frequency_out.View());
    auto cor = forward.CreateFrequencyDomainHostImage();
    forward.CalculateAndNormalize(corView, cor.View());
    auto GIn = Normalize(Hadamard(frequencyIn.View(),cor.View()));
    inverse.CalculateAndNormalize(GIn, outView);
    HostImage<float, 2> o(size_res);
    auto freq = o.View();
    Copy(frequency_out.View(), o);
    Copy(frequencyOut.View(), out);*/


}
}
    }
    }
