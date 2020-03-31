// Copyright 2017 Eyen SE
// Author: Tomas Krupka tomas.krupka@eyen.se

#pragma once

#include <string>

namespace bolt {

/// Save contents of a view as a raw file importable by ImageJ
/// All combinations of Host/Device/Texture, MemoryBased/Procedural
/// views are supported.
/// \param view Input view
/// \param prefix Filename prefix
/// \return Filename of the stored file
template<typename TView>
std::string dump(TView view, std::string prefix);

/// Load image from a file saved using the function Dump.
/// The data is loaded to a view provided as a parameter.
/// All combinations of Host/Device/Texture, MemoryBased/Procedural
/// views are supported as long as they are write enabled.
/// \param view Output view
/// \param prefix Filename prefix
template<typename TView>
void load(TView view, std::string prefix);

/// Load image from a file saved using the function Dump.
/// The template parameter specifies the type of the returned image
/// and can be any of Host/Device/Texture image. Size has to be
/// specified.
/// \param size Image size
/// \param prefix Filename prefix
/// \return Image of type TImage containing data specified by prefix
template<typename TImage>
TImage load(typename TImage::SizeType size, std::string prefix);

/// Filename format: prefix_(ELEMENT_ID)_SIZE[0]xSIZE[1]x...xSIZE[k].raw
/// ELEMENT_ID = (size of element in bits)(TYPE_ID)(CHANNELS)
/// TYPE_ID = getIdentifier(Type)
/// CHANNELS = empty or C(number of channels)
///   View<Float3, 2>(64, 128) : prefix_32FC3_64x128.raw
///   View<double, 2>(64, 128) : prefix_64D_64x128.raw

}  // namespace bolt

#include "image_io.tcc"
