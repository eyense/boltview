
#include <QLabel>
#include <string>
#pragma once

/// Starts running anisotropic diffusion with setted parameters
/// \param input_image Input image, can be read only
/// \param output_image Output image - must provide write access
/// \param iterations_number Number of complete iterations
/// \param coefficient Kappa - Conduction coefficient, e.g. 20-100.
/// Kappa controls conduction as a function of the gradient.
/// If kappa is low small intensity gradients are able to block conduction and hence diffusion across steep edges.
/// A large value reduces the influence of intensity gradients on conduction.
/// \param speed Gamma - controls speed of diffusion. Pick a value <=.25 for stability.
void AnisotropicDiffusion(QImage & input_image, QImage & output_image, int iterations_number, int coefficient, float speed);


/// \return Computes the pixel's weight from the anisotropic diffusion equation
/// \param coefficient Kappa - Conduction coefficient, e.g. 20-100.
/// Kappa controls conduction as a function of the gradient.
/// If kappa is low small intensity gradients are able to block conduction and hence diffusion across steep edges.
/// A large value reduces the influence of intensity gradients on conduction.
/// Pixel diffusions calls of \param callable for each index in <from, to> interval
template<typename type, typename TCallable, typename TLocator>
float PixelDiffusion( int coefficient, type from, type to,  TCallable callable, TLocator locator);


template<typename TConvolutionKernel, typename TView>
void StartAnisotropicDiffusion(TView & input_view, TView & outut_view, const TConvolutionKernel & kernel, int coefficient, float speed);


