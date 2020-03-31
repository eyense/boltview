#include "thresholding.h"
#include <eyen/core/math.h>
#include "ecip/host_image.h"
#include "ecip/convolution_kernels.h"
#include "ecip/convolution.h"
#include "ecip/image_io.h"
#include "ecip/host_image_view.h"
#include <boltview/transform.h>
#include "process_qimage.h"

using namespace eyen;
using namespace ecip;

struct ToBinaryFunctor {
    float _value;
    ToBinaryFunctor(float T) :_value(T){}
    float operator ()(float val) {
        return val > _value ? 255 : 0;
     }
};


void Thresholding (QImage & input_image, float value) {
    auto input_image_view = ToView(input_image);
    auto output_image_view = input_image_view;
    Transform(input_image_view, output_image_view, ToBinaryFunctor{(value)});
    Copy(output_image_view, input_image_view);
}

