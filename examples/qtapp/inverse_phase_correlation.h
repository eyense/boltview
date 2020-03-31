#pragma once

#include <QImage>

namespace eyen {
namespace ecip {
namespace experimental {
    void InversePhaseCorrelation(QImage & im, QImage &correlation_res, QImage & out_im, int x, int y, int size_w, int size_h);
}
}
}

