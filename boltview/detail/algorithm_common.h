#pragma once

#include <boltview/image_locator.h>
#include <boltview/detail/shared_memory_preload.h>


#ifdef BOLT_USE_THREAD_BUILDING_BLOCKS
#include <tbb/tbb.h>
#endif

namespace bolt {

namespace detail {
#if defined(__CUDACC__)

template <typename TView_In, typename TView_Out, typename TFunctor, typename TPolicy, typename TLocator>
BOLT_GLOBAL void
kernelIteration(const TView_In view, const TView_Out view_out, const TFunctor functor, const TPolicy policy)
{
	const int pointCount = policy.pointCountPerKernel(view);
	const auto coordOrig = mapBlockIdxAndThreadIdxToViewCoordinates<DataDimension<TView_In>::value>();
	const auto extents = dataSize(view);

	for (int point = 0; point < pointCount; ++point) {
		const auto coord = policy.pointCoordinates(view_out, coordOrig, point);
		auto locator = TLocator::create(view, coord, policy);
		if (coord < extents) {
			functor(locator, coord);
		}
	}
}

template <typename TView_In, typename TView_Out, typename TFunctor, typename TPolicy, int pointCount, typename TLocator>
BOLT_GLOBAL void
kernelIterationN(const TView_In view, const TView_Out view_out, const TFunctor functor, const TPolicy policy)
{
	const auto coordOrig = mapBlockIdxAndThreadIdxToViewCoordinates<DataDimension<TView_In>::value>();
	const auto extents = dataSize(view_out);

	#pragma unroll pointCount
	for (int point = 0; point < pointCount; ++point) {
		const auto coord = policy.pointCoordinates(view_out, coordOrig, point);
		auto locator = TLocator::create(view, coord, policy);
		if (coord < extents) {
			functor(locator, coord);
		}
	}
}

template <typename TView_In, typename TView_Out, typename TFunctor, typename TPolicy, typename TLocator>
BOLT_GLOBAL void
kernelIteration1(const TView_In view, const TView_Out view_out, const TFunctor functor, const TPolicy policy)
{
	const auto coord = mapBlockIdxAndThreadIdxToViewCoordinates<DataDimension<TView_In>::value>();
	const auto extents = dataSize(view_out);
	auto locator = TLocator::create(view, coord, policy);
	if (coord < extents)
	{
		functor(locator, coord);
	}
}

#endif // __CUDACC__

template<typename TView>
struct ViewIndexingLocator
{
	TView view_;
	typename TView::IndexType location_;
public:
	// Upgrade to CUDA 11.4 broke this and the constructor was not called from the create() method.
	// It is no longer needed as we can leave it up to default construction from initializer list.
	/* BOLT_HD_WARNING_DISABLE BOLT_DECL_HYBRID
	ViewIndexingLocator(const TView & view, const typename TView::IndexType & location):
		view_(view),
		location_(location){};*/

	using AccessType = decltype(view_[typename TView::IndexType{}]);

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID AccessType get() const
	{
		return view_[location_];
	}
        BOLT_HD_WARNING_DISABLE
	template<typename TPolicy>
	static BOLT_DECL_HYBRID ViewIndexingLocator create(const TView & view, const typename TView::IndexType &location, const TPolicy &  /*policy*/)
	{
		return ViewIndexingLocator{view, location};
	}
};

template <typename TView, bool tSharedMemoryPreload>
struct LocatorConstructorImageViewImpl
{
//TODO(johny) - find a better way
#if defined(__CUDACC__)
	using ViewType = DeviceImageConstView< const typename TView::Element, TView::kDimension>;
	template<typename TPolicy>
	static BOLT_DECL_DEVICE  ImageLocator<ViewType, BorderHandlingTraits<BorderHandling::kNone>> create(const TView & view, const Vector<int, TView::kDimension> & location, const TPolicy & policy)
	{
		extern __shared__ char d[]; //TODO(johny) - check how it works with additional shared memory varibles in the functor
		typename TView::Element *data = reinterpret_cast<typename TView::Element *>(d);
		loadToSharedMemory(view, policy, data);
		__syncthreads();
		auto size = getSharedMemoryViewSize(policy.overlapStart + policy.overlapEnd);
		auto tmp_view = makeViewForSharedMemoryBuffer(data, size, stridesFromSize(size));
		auto tmp_view_coords = getViewCoordsInBlock<TView::kDimension>();
		auto locator = LocatorConstruction<BorderHandling::kNone>::create(tmp_view, tmp_view_coords + policy.overlapStart);
		return locator;
	}
#endif // __CUDACC__
};

template <typename TView>
struct LocatorConstructorImageViewImpl<TView, false>
{
	template<typename TPolicy>
	static BOLT_DECL_HYBRID ImageLocator<TView, BorderHandlingTraits<TPolicy::kBorderHandling>> create(const TView & view, const Vector<int, TView::kDimension> & location, const TPolicy &  /*policy*/)
	{
		auto locator = LocatorConstruction<TPolicy::kBorderHandling>::create(view, location);
		return locator;
	}
};

template <typename TView, bool tSharedMemoryPreload>
struct LocatorConstructor {

	template<typename TPolicy>
	static BOLT_DECL_HYBRID
	auto create(const TView & view, const Vector<int, TView::kDimension> & location, const TPolicy & policy)
	{
		return LocatorConstructorImageViewImpl<TView, tSharedMemoryPreload>::create(view, location, policy);
	}
};

#ifdef BOLT_USE_THREAD_BUILDING_BLOCKS
template< typename TView, typename TFunctor, typename TLocator, typename TPolicy>
class ApplyFunctor {
	TView view_;
	mutable TFunctor functor_;
	TPolicy policy_;
public:
	void operator()( const tbb::blocked_range<size_t>& r ) const {
		for (size_t i = r.begin(); i != r.end(); ++i)
		{
			auto index = getIndexFromLinearAccessIndex(view_, i);
			auto locator = TLocator::create(view_, index, policy_);
			functor_(locator, index);
		}
	}
	ApplyFunctor(const TView & view, const TFunctor & functor, const TPolicy & policy ) :
		view_(view),
		functor_(functor),
		policy_(policy)
    {}
};
#endif

#ifdef BOLT_USE_THREAD_BUILDING_BLOCKS
template <typename TView, typename TFunctor, typename TPolicy,typename TLocator>
typename std::enable_if<TPolicy::bParallelHost>::type
iterateHost(TView view, TFunctor functor, TPolicy policy, int i)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0,view.elementCount()),  ApplyFunctor<TView, TFunctor, TLocator, TPolicy>(view, functor, policy));
}
#endif

template <typename TView, typename TFunctor, typename TPolicy,typename TLocator>
void
iterateHost(const TView & view, const TFunctor & functor, const TPolicy & policy, ...)
{

	for (int64_t i = 0; i < view.elementCount(); ++i) {
		auto index = getIndexFromLinearAccessIndex(view, i);
		auto locator = TLocator::create(view, index, policy);
		functor(locator, index);
	}
}

#if defined(__CUDACC__)
template<int tDimension>
inline Vector<int, tDimension> getPreloadSize(dim3 blockSize, Vector<int, tDimension> overlap){
	return dim3ToInt<tDimension>(blockSize) + overlap;
}

template <typename TInView>
struct SharedMem
{
	template <typename TPolicy>
	static typename std::enable_if<TPolicy::kPreloadToSharedMemory, int>::type RunSharedMemoryCheck(dim3* blockSize, TPolicy* policy)
	{
		auto size = getPreloadSize<TInView::kDimension>(*blockSize, policy->overlapStart + policy->overlapEnd);
		int sharedMemorySize = product(size) * sizeof(typename TInView::Element);

		int device;
		cudaGetDevice(&device);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);

		if (sharedMemorySize > deviceProp.sharedMemPerBlock)
		{
			BOLT_THROW(CudaError() << MessageErrorInfo("Cannot fit into shared memory."));
		}

		return sharedMemorySize;
	}

	static int RunSharedMemoryCheck(...)
	{
		return 0;
	}
};
#endif

template <bool tRunOnDevice, typename TLocator>
struct IterateImplementation {

#if defined(__CUDACC__)
	template <typename TView_In, typename TView_Out, typename TFunctor, typename TPolicy>
	static void run(const TView_In & view, const TView_Out & view_out, const TFunctor & functor, const TPolicy & policy, const cudaStream_t & cuda_stream) {
		dim3 blockSize = policy.blockSize();
		dim3 gridSize = policy.gridCount(view_out);

		int sharedMemorySize = SharedMem<TView_Out>::RunSharedMemoryCheck(&blockSize, &policy);

		const int pointCount = policy.pointCountPerKernel(view_out);

		// solves crash in DeviceTricubicInterpolationTest
		// the test runs out of registers otherwise
                // times on my laptop DynamicDeviceTestPerf from convolution_test: 0.00407 +/- 0.0020 for this version, 0.00566 +/- 0.0024 for for loop
                // Perhaps more performance testing will be needed....
		if ( pointCount <= 1)
		{
			detail::kernelIteration1<TView_In, TView_Out, TFunctor, TPolicy, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else if ( pointCount <= 2)
		{
			detail::kernelIterationN<TView_In, TView_Out, TFunctor, TPolicy, 2, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else if ( pointCount <= 3)
		{
			detail::kernelIterationN<TView_In, TView_Out, TFunctor, TPolicy, 3, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else if ( pointCount <= 4)
		{
			detail::kernelIterationN<TView_In, TView_Out, TFunctor, TPolicy, 4, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else
		if ( pointCount <= 5)
		{
			detail::kernelIterationN<TView_In, TView_Out, TFunctor, TPolicy, 5, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else if ( pointCount <= 7)
		{
			detail::kernelIterationN<TView_In, TView_Out, TFunctor, TPolicy, 7, TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		else
		{
			detail::kernelIteration<TView_In, TView_Out, TFunctor, TPolicy,TLocator>
				<<< gridSize, blockSize, sharedMemorySize, cuda_stream>>>(view, view_out, functor, policy);
		}
		BOLT_CHECK_ERROR_STATE("kernelIteration");
	}
#endif  // __CUDACC__
};

template<typename TLocator>
struct IterateImplementation<false, TLocator> {

	template <typename TView_In, typename TView_Out, typename TFunctor, typename TPolicy>
	static void run(const TView_In & view, const TView_Out & view_out, const TFunctor & functor, const TPolicy & policy, const cudaStream_t &  /*cuda_stream*/) {
		int i= 0; //Used only for overload resolution
		detail::iterateHost<TView_In, TFunctor, TPolicy, TLocator>(view, functor, policy, i);
	}
};

} // namespace detail
} // namespace bolt

