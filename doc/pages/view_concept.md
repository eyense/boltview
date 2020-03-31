# View Concept {#view_concept}

BoltView is designed in a way that all accesses to image data is done through image views. Image views are lightweight memcopyable non-owning containers. They are used in similar manner as iterators in STL. 
Since iterators themselves are not ideal solution in parallel environment, data views were chosen in BoltView implementation as the most suitable replacement. 

Image view concept is bound to a notion of zero based n-dimensional interval. For each point from the views n-d interval we get access to image element (whether read or read/write depends on type of view).

Available views are listed in \ref Views "Views Module"

## Host Image View
Host image views are views returned either by host image instances or by generating subview from already available host view. 

## Device Image View
Similar to host image views - created either by device image or by getting subview of already created device image view.

We refer to both Host and Device Image views memory based views.

## Procedural View

Procedural views are quite different from memory based views as they do not provide a direct memory access, Instead, they either generate the value based on their arguments and provided index or do some on demand operation on a wrapped view of different type (see examples). These views are lazily evaluated - returned value is not computed until the position is accessed.

In lots of situations we can achieve the same result by using meta-algorithms to compute some output view data. But if our algorithm has several steps it would mean several kernel executions. 
By using lazy views, we can bundle all required operations in one procedural view and get the result in single kernel execution, thus lowering the overhead.

A model situation when this approach pays off is image algebra.

