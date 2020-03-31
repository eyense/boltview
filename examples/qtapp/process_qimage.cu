#include "process_qimage.h"
#include "ecip/copy.h"
using namespace eyen;
using namespace ecip;


HostImageView<uchar, 2> ToView(QImage& image) {
    auto image_bits_iterator = image.bits();
    const Int2 size_of_image(image.size().width(), image.size().height());
    return MakeHostImageView(image_bits_iterator, size_of_image);
}


HostImageView<float, 2> ToFloatView(HostImageView<uchar, 2> & view ,Int2 size) {
    HostImage<float, 2> result_image(size);
    auto result_view = result_image.View();
    Copy(view, result_view);
    return result_view;
}

