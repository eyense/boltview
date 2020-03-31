# Fractals {#fractals}

<p float="left" align="center">
<img src="mandelbrot.jpg" alt="Mandelbrot set" title="Mandelbrot set" width="300">
<img src="mandelbrot2.jpg" alt="Mandelbrot set detail" title="Mandelbrot set detail" width="300">
</p>

In this article we will explore the basic usage of the BoltView concepts to generate image of the famous Mandelbrot's set fractal. We will start with simple version running on CPU -- version useful for algorithm debugging. After that we will modify the code to run on GPU.

The first step is to prepare a place to store the actual image data. For this purpose we will create an instance of `bolt::HostImage` template. It is a class following the RAII (Resource Aquisition Is Initialization) paradigm. The two template parameters are the type of pixel values (`Vector<uint8_t, 3>` -- to represent RGB color in 8-bit precision) and dimension of the image. Constructor takes the actual size of the allocated image.

```
bolt::HostImage<Vector<uint8_t, 3>, 2> output_image(Int2(3500, 2000));
```

The computation of the actual value of each pixel is independent from each other. That means we can use meta algorithm from the for-each family to execute a callable for each pixel returning a computed value. This callable would need to know not only where to store the value, but also its position in the image so the right variant of the meta-algorithm is `bolt::forEachPosition`. It takes an *image view* which provides write access to all the image pixels and a callable that takes a reference to a pixel together with its n-D index. 

```
bolt::forEachPosition(view(output_image), Mandelbrot{output_image.size()});
```
In the above snipped the `view()` call creates the view for the output image with the right access and the `Mandelbrot{}` is a functor wrapping the computation.

```
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
