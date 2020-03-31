
#pragma once

namespace bolt {

namespace detail {

#ifdef __CUDACC__
template<int tDimension>
BOLT_DECL_HYBRID
dim3 defaultBlockDimForDimension();

template<>
BOLT_DECL_HYBRID
inline dim3 defaultBlockDimForDimension<1>() { return dim3(512, 1, 1); }

template<>
BOLT_DECL_HYBRID
inline dim3 defaultBlockDimForDimension<2>() { return dim3(64, 8, 1); }

template<>
BOLT_DECL_HYBRID
inline dim3 defaultBlockDimForDimension<3>() { return dim3(32, 4, 4); }

BOLT_DECL_HYBRID
constexpr int defaultNumThreadsPerBlock() { return 512; }

template<int tDimension>
BOLT_DECL_HYBRID
dim3 defaultGridSizeForBlockDim(Vector<int, tDimension> view_dimensions, dim3 block_size);

template<>
BOLT_DECL_HYBRID
inline dim3
defaultGridSizeForBlockDim<1>(Vector<int, 1> view_dimensions, dim3 block_size)
{
	return dim3(
		(view_dimensions[0] - 1) / block_size.x + 1,
		1,
		1);
}

template<>
BOLT_DECL_HYBRID
inline dim3
defaultGridSizeForBlockDim<2>(Vector<int, 2> view_dimensions, dim3 block_size)
{
	return dim3(
		(view_dimensions[0] - 1) / block_size.x + 1,
		(view_dimensions[1] - 1) / block_size.y + 1,
		1);
}

template<>
BOLT_DECL_HYBRID
inline dim3
defaultGridSizeForBlockDim<3>(Vector<int, 3> view_dimensions, dim3 block_size)
{
	return dim3(
		(view_dimensions[0] - 1) / block_size.x + 1,
		(view_dimensions[1] - 1) / block_size.y + 1,
		(view_dimensions[2] - 1) / block_size.z + 1);
}

inline BOLT_DECL_HYBRID dim3 defaultMaxGridSize()
{
	return dim3(2147483647, 65535, 65535); //TODO
}

inline BOLT_DECL_HYBRID size_t divideRoundUp(size_t numerator, size_t denominator)
{
	return (numerator + denominator - 1)/denominator;
}

inline BOLT_DECL_HYBRID dim3 defaultIterationCountsPerKernel(dim3 gridCount, dim3 maxGridSize)
{
	return dim3(divideRoundUp(gridCount.x,maxGridSize.x), divideRoundUp(gridCount.y,maxGridSize.y), divideRoundUp(gridCount.z,maxGridSize.z));
}

inline BOLT_DECL_HYBRID int defaultIterationCountPerKernel(dim3 iterationCounts)
{
	return iterationCounts.x * iterationCounts.y * iterationCounts.z;
}

inline BOLT_DECL_HYBRID Vector<int, 1> defaultPointCoordinates( const Vector<int, 1>& base, int index, dim3 maxGridSize, dim3 blockSize)
{
	return Vector<int, 1>(base[0]+ index * maxGridSize.x * blockSize.x);
}

inline BOLT_DECL_HYBRID Vector<int, 2> defaultPointCoordinates(const Vector<int, 2>& base, int index, dim3 maxGridSize, dim3 blockSize, dim3 iterationsCount)
{
	const unsigned int xPos = index % iterationsCount.x;
	const unsigned int yPos = index / iterationsCount.x;
	return Vector<int, 2>(base[0]+ xPos * maxGridSize.x * blockSize.x, base[1] + yPos * maxGridSize.y * blockSize.y);
}

inline BOLT_DECL_HYBRID Vector<int, 3> defaultPointCoordinates(const Vector<int, 3> &base, int index, dim3 maxGridSize, dim3 blockSize, dim3 iterationsCount)
{
	const unsigned int xPos = index % iterationsCount.x;
	const unsigned int yPos = index / iterationsCount.x % iterationsCount.y;
	const unsigned int zPos = index / iterationsCount.x / iterationsCount.y;
	return Vector<int, 3>(base[0] + xPos * maxGridSize.x * blockSize.x, base[1] + yPos * maxGridSize.y * blockSize.y,
						  base[2] + zPos * maxGridSize.z * blockSize.z);
}

inline BOLT_DECL_HYBRID dim3 defaultKernelThreadGeometry(dim3 gridCount, dim3 maxSize)
{
	return dim3(bolt::min(gridCount.x, maxSize.x), bolt::min(gridCount.y, maxSize.y), bolt::min(gridCount.z, maxSize.z));
}

template<bool tIsDevice, bool tIsHost>
struct AvailableExecutions {
	static constexpr bool kIsDevice = tIsDevice;
	static constexpr bool kIsHost = tIsHost;
};

template<typename TView>
struct AvailableExecutionsForView {
	using type = AvailableExecutions<TView::kIsDeviceView, TView::kIsHostView>;
};

template<typename TAvailableExecutions1, typename TAvailableExecutions2>
struct CombineAvailableExecutions {
	using type = AvailableExecutions<
			TAvailableExecutions1::kIsDevice && TAvailableExecutions1::kIsDevice,
			TAvailableExecutions1::kIsHost && TAvailableExecutions1::kIsHost>;
};

#endif  // __CUDACC__
}  // namespace detail

}  // namespace bolt
