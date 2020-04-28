# Fractals {#fractals}

<p float="left" align="center">
<img src="mandelbrot.jpg" alt="Mandelbrot set" title="Mandelbrot set" width="300">
<img src="mandelbrot2.jpg" alt="Mandelbrot set detail" title="Mandelbrot set detail" width="300">
</p>

In this article we will explore the basic usage of the BoltView concepts to generate image of the famous Mandelbrot's set fractal. We will start with simple version running on CPU -- version useful for algorithm debugging. After that we will modify the code to run on GPU.

The first step is to prepare a place to store the actual image data. For this purpose we will create an instance of `bolt::HostImage` template. It is a class following the RAII (Resource Aquisition Is Initialization) paradigm. The two template parameters are the type of pixel values (`Vector<uint8_t, 3>` -- to represent RGB color in 8-bit precision per channel) and dimension of the image. Constructor takes the actual size of the allocated image.

```c++
bolt::HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(3500, 2000));
```

The computation of the actual value of each pixel is independent from each other. That means we can use meta algorithm from the for-each family to execute a callable for each pixel returning a computed value. This callable would need to know not only where to store the value, but also its position in the image so the right variant of the meta-algorithm is `bolt::forEachPosition`. It takes an *image view* which provides write access to all the image pixels and a callable that takes a reference to a pixel together with its n-D index. 

```cpp
bolt::forEachPosition(view(output_image), Mandelbrot{output_image.size()});
```
In the above snipped the `view()` call creates the view for the output image with the right access and the `Mandelbrot{}` is a functor wrapping the computation.

```c++
struct Mandelbrot
{
    static constexpr int max_iteration = 500;
    Mandelbrot(
        Int2 aExtents,
        Region<2, float> r = { Float2(-2.5f, -1.0f), Float2(3.5f, 2.0f) }):
            extents(aExtents),
            region(r)
    {}

    static BOLT_DECL_HYBRID Vector<uint8_t, 3>
    colorMap(int iteration) {
        Vector<uint8_t, 3> tmp;
        tmp[0] = iteration % 256;
        tmp[1] = (iteration * 7) % 256;
        tmp[2] = (iteration * 13) % 256;
        return tmp;
    }

    template<typename TValue>
    BOLT_DECL_HYBRID void
    operator()(TValue &val, Int2 position) const {
        auto coords = product(div(Float2(position), extents), region.size) + region.corner;

        HostComplexType z0{ coords[0], coords[1] };
        HostComplexType z{0, 0};

        int iteration = 0;
        while (magSquared(z) < 4  &&  (iteration < max_iteration)) {
            z = z*z  + z0;
            ++iteration;
        }
        val = colorMap(iteration);
    }
    Int2 extents;
    Region<2, float> region;
};
```

The functor’s constructor takes two parameters, which specify the mapping between the pixels and the actual set’s domain, so you can zoom in any part of the Mandelbrot’s set. Next is a static method `colorMap()`, which just returns some random color for every integer — we use it for the set coloring. Notice that the declaration is prefixed by `BOLT_DECL_HYBRID` this marks the method as callable on both device and host.

And the final block in the functor is the actual operator() overload, which takes reference to the pixel it will be computing and its position in the image. The implementation is straightforward — map the pixel coordinates to the set’s domain and find the iteration when the expression core expression diverges and color map the iteration number.

Now we have the complete CPU version working. But how do we run the computation on GPU?
It is easy — the computation is done by the call to `bolt::forEachPosition()`, the algorithm chooses where to run based on the type of the image view, which is passed as the first argument. We passed a host image view, so the execution was on CPU, but if we simply use the device image to store our results, it will be executed on GPU.
```c++
bolt::DeviceImage<Vector<uint8_t, 3>, 2> device_image(Int2(3500, 2000));
bolt::forEachPosition(
        view(device_image), 
        Mandelbrot{device_image.size()});
```

And that is all it takes to run our code on GPU. Only few small notes at the end:
 * The execution on GPU is asynchronous.
 * The result image lies on GPU, so it is not accessible from host code until you copy it to host memory, or use unified memory image, which is usable on both host and device.
 * Every function, which could run on GPU must be annotated by `BOLD_DECL_DEVICE` or `BOLT_DECL_HYBRID` (those are just macros wrapping CUDA `__device__` and `__device__ __host__` pragmas).
