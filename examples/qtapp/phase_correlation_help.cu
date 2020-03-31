#include "phase_correlation_help.h"
#include "inverse_phase_correlation.h"
using namespace eyen;
using namespace ecip;
using namespace experimental;
void InverseCorrelation(QImage &im, QImage &cor, QImage & out,  int x, int y, int size_w, int size_h) {
    InversePhaseCorrelation(im, cor, out, x, y, size_w, size_h);
}


