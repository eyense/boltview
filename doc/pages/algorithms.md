# Algorithms {#algorithms}

[TOC]

Several STL-like meta algorithms are provided, hiding kernel (or several kernels) execution for common computation patterns. This is similar to what Thrust does for 1D data structures.

## Algorithm Execution

Most of the algorithms from the BoltView library are templated by the type of the image view(s), which are passed as arguments. Since all the image view types contain an information on how the image elements can be accessed (from device, host, or both) the algorithm implementation can choose in compile time a proper way of execution, either on device or on host. If execution is possible on both device and host the device variant is preferred.

## For Each

First class of meta-algorithms is the for-each family. For-each algorithms execute a provided functor for all image elements accessible from the image view, which is passed as a first parameter. They differ in what kind of additional information is passed to the functor's `operator()`. 

### bolt::forEach()

The basic for-each variant executes the provided functor on a references to image element values. It means that if the input image view provide write access then the reference is non-const and the functor can update the element value. In case of read-only image view access (const image views, procedural views, etc.) the reference will be const and thus the functor cannot change the elements value.

### bolt::forEachPosition()

Second variant of the for-each algorithm, which works exactly as the basic variant, only difference is that it also passes the n-D  index of the image element as the second parameter to the functor.

### bolt::forEachLocator()

The third for-each variant does not directly provide the value reference, but instead it creates an instance of image locator. Image locator can be viewed as an n-D extension of iterator, which is anchored to the image element position and allows access to the nearest element neighborhood.

**WARNING:** if you update the element value through the image locator it can introduce data-race condition, when accessing the neighborhood as the neighborhood value can be the original one or the updated one depending on the timing. 

## Transform

Transform algorithm family extends the concept introduced by the for-each meta-algorithms by adding second image view argument, which is used for storing the results of the functor executions. 
Because of that the main requirement is the need for identical size of both input and output image views.

### bolt::transform()

Basic variant - each element is passed to the functor and the result is stored under the exact samer index in the output image view.

### bolt::transformPosition()

Extends the basic behavior by passing n-D element index as the second parameter for the functor execution.

### bolt::transformLocator()

The access to the input image views is in the form of an image locator. Since the input is not changed in this case, the data-race condition situation possible in the modifiing `forEachLocator` is prevented.

This algorithm is useful for spatial filter implementations. Good example is the `convolution()` algorithm, which uses it internally.

## Copy and CopyAsync

`bolt::copy` and `bolt::copyAsync` meta-algorithms generally serve for copying data between views of the same size, but there are some restrictions:
 - When copying between device and host based views, only memory based views can be used. 
 - Procedural and lazily evaluated views can only be copied only in device-device or host-host transactions.

## Convolution

## Reductions

### bolt::reduce()

### bolt::dimensionReduce()

## FFT

Module for *Fast Fourier Transformation* provides utilities, which simplifies usage of CUFFT library for FFT computation on device and libfftw for execution on host.

Both librariers have a similar design in which the first required step is to create a *plan*. Plan wraps the basic computation parameters (input/output size, algorithm variant, etc.) and memory for intermediate results used internaly by the algorithms.

