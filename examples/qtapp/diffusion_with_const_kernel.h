#include <QLabel>
#include <string>
#include <eyen/core/math.h>
#include <iostream>
#pragma once



/// Applies \param iterations_number iterations of Diffusion with constant kernel
/// \param input_image Input image, can be read only
/// \param output_image Output image - must provide write access
void DiffusionWithConstKernel (QImage & input_image, QImage & output_image, int iterations_number);

template<typename TView>
void StartDiffusionWithConstKernel (TView & input_view, TView & output_view, int iterations_number);

