#include "anisotropic.h"

#pragma once


/// Applies Sobel edge detection on the \param input_image image
/// Result will be saved to the \param output_image image
void EdgeDetection(QImage &input_image, QImage &output_image);


template<typename TView>
void ConvolutionView(TView & input_view, TView &output_view, float mat[]);

