# Quick Start {#quick_start}

[TOC]

## Install


## Examples

### Example 1
Simple usage pattern - allocate 50^3 integer image in RAM and the same on GPU, copy from host to device, do processing and copy back.

```
// Allocate image in RAM of specified size
HostImage<int, 3> host_image(50, 50, 50);
// Fill host_image (load from file, etc.)
...
// Create image on GPU with same size as host_image
DeviceImage<int, 3> device_image(host_image.Size());

// Copy data from host_image to device_image
copy(host_image.ConstView(), device_image.View());

// Do processing on GPU - replace each image element by its squared value.
transform(device_image.ConstView(), device_image.View(), SquareFunctor()); 

// Copy result back to the host_image
copy(device_image.ConstView(), host_image.View());
```

### Example 2 

Computing SSD (Sum of Squared Differences) of two different checkerboards (procedural views).

No image allocated - all view values generated on the fly when needed.

```
auto view1 = makeCheckerBoardDeviceImageView(1, 0, Int2(2, 2), Int2(16, 16));
auto view2 = makeCheckerBoardDeviceImageView(2, 1, Int2(8, 8), Int2(16, 16));

int ssd = reduce(square(subtract(view1, view2)), 0, thrust::plus<int>());
```


### Example 4 

Wrapping memory buffer as a constant host image view.

```
std::vector<int> buffer(200);
auto view = makeConstHostImageView(buffer.data(), Int2(10, 20), StridesFromSize(Int2(10, 20)));
DeviceImage<int, 2> device_image(view.Size());
copy(view, device_image.View())
```

### Example 4 

Computing correlation in fourier spectrum.

```
// Load image and searched pattern
DeviceImage<float, 3> image = ...;
DeviceImage<float, 3> pattern = ...;

// Pad the pattern image so the FFT spectrum has the same size as the image.
// FFT needs memory buffers, so we need to copy lazily evaluated paddedView to device image.
DeviceImage<float, 3> padded_pattern(image1.Size());
Int3 offset = -Div(pattern.Size(), Int3(2, 2, 2));
copy(paddedView(pattern.View(), image.Size(), offset, 0.0f), padded_pattern.View());

// Allocate images for FFT spectra
DeviceImage<cufftComplex, 3> image_fft(GetFFTImageSize(image.Size()));
DeviceImage<cufftComplex, 3> pattern_fft(GetFFTImageSize(image.Size()));

// Perform forward FFT
FFTCalculator fft_calculator(image.Size(), image.Strides(), image_fft.Size(), image_fft.Strides());
fft_calculator.Forward(image.View(), image_fft.View());

// Compute correlation (complex conjugate multiplication of spectra)
DeviceImage<cufftComplex, 3> tmp_fft(image_fft.Size());
FFTCorrelation(image_fft.ConstView(), pattern_fft.ConstView(), tmp_fft.View());
ECIP_CHECK(cudaThreadSynchronize());

// Inverse FFT to get the result
DeviceImage<float, 3> result(image.Size());
fft_calculator.Inverse(tmp_fft.View(), result.View());
ECIP_CHECK(cudaThreadSynchronize());
// FFT is without normalization factor - apply now.
copy(MultiplyByFactor(1.0f / result.ConstView().ElementCount(), result.ConstView()), result.View());
```

### Example 5 

Compute symmetric difference in X axis (approximation of partial derivative).

```
struct XSymmetricDifferenceFunctor {
    template<typename TLocator>
    ECIP_DECL_DEVICE typename TLocator::AccessType
    operator()(TLocator locator) {
        return (locator.DimOffset<0>(1) - locator.DimOffset<0>(-1)) / 2;
    }
};
...

auto view = MakeCheckerBoardDeviceImageView(2.0f, 0.0f, Int2(2, 2), Int2(16, 16));;
DeviceImage<float, 2> device_image(view.Size());
transformLocator(view, device_image.View(), XSymmetricDifferenceFunctor());
```

### Example 6 

Use image view in custom kernel.

```
template<typename TView>
ECIP_GLOBAL void testKernel(TView view) {
	// Do 1 to 1 mapping between view elements and CUDA threads.
	auto coords = MapBlockIdxAndThreadIdxToViewCoordinates<TView::kDimension>();
	if (coors < view.Size()) {
		// Do some operation on the view element
		view[coords] = view[coords] * Sum(coords);
	}
};

template<typename TView>
void runTestKernel(TView view) {
	dim3 blockSize(16, 16, 1);
	// Setup the grid so each thread process single view element
	dim3 gridSize = DefaultGridSizeForBlockDim(view.Size(), blockSize);
	testKernel<TView><<<gridSize, blockSize>>>(view);
	ECIP_CHECK(cudaThreadSynchronize());
}
```
