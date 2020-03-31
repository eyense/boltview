#include <QImage>
#include "ecip/host_image.h"
#include "ecip/host_image_view.h"
#pragma once
using namespace eyen;
using namespace ecip;

/// Creates image view from \param image QImage
HostImageView<uchar, 2> ToView(QImage& image);




///Functor that returns number in pixel values range
struct ClampFunctor {
    float operator ()(float val) {
        return eyen::Clamp(0.f, 255.f, val);
     }
};


HostImageView<float, 2> ToFloatView(HostImageView<uchar, 2> & view, Int2 size);

