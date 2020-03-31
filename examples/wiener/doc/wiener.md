# Wiener Filtering {#wiener}

In this example we will show how to implement basic version of the Wiener filter to deconvolve a blurred image.

<p float="left" align="center">
<img src="./blurred_bee.png" alt="Blurred image" title="Blurred image" height="300">
<img src="./deconvolved_bee.jpg" alt="Deconvolved image" title="Deconvolved image" height="300">
</p>

Wiener filter is applied in frequency domain, so the key part is to do the Fourier transformation of the blurred image and also of point spread function (PSF), which caused the image blur.


We can create the wiener filter as a procedural view which provides on demand modulation factors for each frequency. 
Procedural view approach is useful when certain computation do not need to be computed right away and can be bundled to some other calculation. The calculation of each factor in Wiener filter is wrapped in the following functor. We use the constant noise suppression factor for all frequencies.
```
struct WienerFtor {

	template<typename TValue>
	BOLT_DECL_HYBRID
	auto operator()(const TValue &val) const {
		return conjugate(val) / (magSquared(val) + noise_factor);
	}

	float noise_factor = 0.0f;
};
```
It is a direct implementation of the Wiener formula. The `operator()` is prefixed with `BOLT_DECL_HYBRID` which causes its implementation to be compiled for both host and device. We use this functor to instance a template `UnaryOperatorImageView<>`, which wraps an image view and pixel access executes the functor on the pixel of the underlying image view.
```
template<typename TPsf>
auto wiener(TPsf psf, float noise_factor) {
	return UnaryOperatorImageView<TPsf, WienerFtor>(psf, WienerFtor{noise_factor});
}
```


Inverse filtering in Fourier domain requires the spectra of the input image and the PSF to be of the same size. This is achieved by padding the PSF by zeroes and keeping the center of the PSF at the coordinates [0, 0] and wrap it cyclically.
```
template<typename TKernel, typename TSize>
BOLT_DECL_HYBRID
auto padConvolutionKernel(TKernel &k, TSize size) {
	auto kernel_view = mirror(makeHostImageConstView(k.pointer(), k.size()), Bool2(true, true));
	auto new_center = k.size() - k .center();
	return paddedView(kernel_view, size, -new_center + Int2(1, 1), 0);
}
```

<img src="./kernel2.png" alt="PSF" title="PSF" height="100">


