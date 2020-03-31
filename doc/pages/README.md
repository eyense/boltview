Overview {#mainpage}
========
[TOC]

BoltView library aims to serve as a set of tools set for basic image manipulation in CUDA environment. It's purpose is to hide complexity introduced by CUDA API without losing control of memory management, kernel execution, etc. One may look at BoltView is that it adds the abstraction layer to CUDA in a similar way the STL does for C++: it hides the somewhat dangerous low-level layer and adds tools that make writing CUDA code easier, more reliable and reusable.

- Hide the CUDA API complexity
- Error handling using exceptions - strong exception guarantees where possible. (Compared to plain CUDA which uses error codes)
- Full control over GPU memory allocations in a safe way
- Resource Acquisition Is Initialization (RAII) wrappers
- Easy interoperability with other party image processing libraries (succesfuly used with ITK and Numpy for example)


