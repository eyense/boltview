# Edge Detection {#edge_detection}

<p float="left" align="center">
<img src="bee_bw.jpg" alt="Original bee image" title="Original bee image" width="300">
<img src="edge_detection.jpg" alt="Sobel based edge detection" title="Sobel based edge detection" width="300">
</p>

In this example we will implement simple edge detection algorithm based on gradient magnitude computation. We will use convolution with Sobel's kernels to compute the partial derivatives.

One possible approach would be to compute convolution with each of the kernels by calling `convolution()` function and then computing the output edge response from the resulting images. Major disadvantage of this approach is that it would mean multiple passes through the input and intermediate data, which may be expensive especially in case that the input data are stored in unified memory and does not fit completely to the GPU memory - in that case the driver would have to transfer the data between RAM and GPU memory multiple times.

Here we will present different approach - we will compute the edge response directly in custom functor passed to `transformLocator()` algorithm. We are using `transformLocator()` because we need to access 3x3 neighborhood of each pixel to compute the local gradient estimation.

```c++

aaa
```
