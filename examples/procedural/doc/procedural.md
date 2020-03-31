# Procedural Views {#procedural}

<p float="left" align="center">
<img src="./checkerboard_superposition.jpg" alt="Checkerboard superposition" title="Checkerboard superposition" width="300">
</p>

This will be a short presentatation how a procedural image views can be used. We will generate an image which consists of multiple checkerboards with different tile size being superpositioned over each other and summed.

The BoltView library provides an purely procedural image view which behaves like checker board image view, although the checker board image does not exist in memory. The view computes proper value on demand when executing the `operator[]`. The library also provides overloads of arithmetic operators, which can be used for basic manipulation with image views, although the view they return also does not have data representation in memory. A wrapper view containing operation inputs is returned and calculation happens again on demand.

Types of all these procedural views are known in compile time, so we can do a little trick and create a wrapper view which does our checker board summation on demand.

```c++
template<int... tArgs>
auto generate(Int2 size, std::integer_sequence<int, tArgs...> seq) {
	return fold(
		[](auto first, auto second) { return first + second; },
		checkerboard(uint8_t(0), uint8_t(255/seq.size()), Int2(power(3, tArgs), FillTag{}), size)
		...);
}
```
We used a integer sequence to provide arguments for each checkerboard level. Because in CUDA we are currently limited to C++14, we cannot use [fold expressions](https://en.cppreference.com/w/cpp/language/fold) for parameter pack, which are available in C++17, so we use a utility function `fold()` which applies operation passed as the first parameter and applies it to the variadic parameter list like `(((a op b) op c) op d)`.

The presented `generate()` function does not do any calculation - it returns procedural view which does lazy evaluation. To get actual pixel values we need to copy the values into actual image, which we can save as in the following snippet.
```c++
Int2 size{2187, 2187};

auto im = generate(size, std::make_integer_sequence<int, 7>{});

HostImage<uint8_t, 2> output_image(size);
copy(im, view(output_image));
saveImage("output.jpg", constView(output_image));
```

Output of this example is shown at the top of the page.
