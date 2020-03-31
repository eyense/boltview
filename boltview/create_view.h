// Copyright 2019 Eyen SE
// Author: Martin Hora martin.hora@eyen.se

#pragma once

#if 0
/// Create view of an image, which can be used for modification of image data.
template<typename TImage, typename TView, typename TPolicy>
TView view(TImage& image, TPolicy policy);

/// Create view of a whole image, which can be used for modification of image data.
template<typename TImage, typename TPolicy = typename TImage::Policy>
typename TImage::ViewType view(TImage& image, TPolicy = TPolicy()) {
	return image.view();
}

/// Create view of an image, which can be used for const access to the image data.
template<typename TImage, typename TView, typename TPolicy>
TView constView(const TImage& image, TPolicy policy);

/// Create view of a whole image, which can be used for const access to the image data.
template<typename TImage, typename TPolicy = typename TImage::Policy>
typename TImage::ConstViewType constView(const TImage& image, TPolicy = TPolicy()) {
	return image.constView();
}

#endif  // 0

namespace bolt {
/// Create view of a whole image, which can be used for modification of image data.
template<typename TImage>
auto view(TImage& image) {
	return image.view();
}

/// Create view of a whole image, which can be used for const access to the image data.
template<typename TImage>
auto constView(const TImage& image) {
	return image.constView();
}

}  // namespace bolt
