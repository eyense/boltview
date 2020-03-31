# Image Concept {#image_concept}

Image is a container which owns a memory block containing pixel/voxel data. It allocates data in constructor and deallocates in destructor. 
Image data are accessed by provided image views - ConstView() for read only access and View() for read/write access.

There are several types of images based on type of the memory it allocates from. 
All image types are created in the host code (CPU code). In general, CUDA kernels and device code in can only access Imaages through provided image views (DeviceImage, TextureImage). This is because device code cannot access RAM. (Although we are considering supporting Unified Memory in the future).

## Host Image
For data allocated in RAM, the generic bolt::HostImage is provided. It is templated by type of image element (pixel/voxel) and dimension of the image (2D and 3D images are currently supported).
Since data are allocated in the host memory (RAM), they can be easily accessed by the provided image views from host code. You can even access the internal memory buffer directly by pointer access, 
but be aware that image owns the data, so the pointer is invalidated in destructor and possibly during image resize.

## Device Image
Similar to HostImage, but the internal data buffer is allocated as linear device (GPU) memory. Implemented by generic bolt::DeviceImage Because the data live on GPU, it is accessible from device code by provided image views.

## Texture Image
Similar to the Device Image, it contains data accessible from device code, but data are allocated in texture memory and accessed through bindless texture interface (added in CUDA 5.0). Texture memory has different caching properties in commparison to linear memory, 
and also provides additional utilities, such as normalised floating point access, element interpolation and mipmapping.


## Unified Image
