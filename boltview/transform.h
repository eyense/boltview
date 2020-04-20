#pragma once

#include <boltview/detail/algorithm_common.h>
#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/image_locator.h>
#include <boltview/detail/meta_algorithm_utils.h>
#include <boltview/detail/shared_memory_preload.h>

#if defined(__CUDACC__)
#include <boltview/device_image_view.h>
#endif  //defined(__CUDACC__)
namespace bolt {

template<typename TView1, typename TView2>
struct DefaultTransformPolicy {
#if defined(__CUDACC__)
	static BOLT_DECL_HYBRID dim3 blockSize()
	{
		return detail::defaultBlockDimForDimension<TView1::kDimension>();
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID dim3 gridSize(const TView3 &view)
	{
        	return detail::defaultGridSizeForBlockDim(dataSize(view), blockSize());
	}

	static BOLT_DECL_HYBRID dim3 maxGridSize()
	{
		return detail::defaultMaxGridSize();
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID dim3 iterationCounts(const TView3 &view)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 gridCount =  gridSize(view);
		return detail::defaultIterationCountsPerKernel( gridCount, max_grid_size);
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID int pointCountPerKernel(const TView3 &view)
	{
		const dim3 iterationCount = iterationCounts(view);
		return detail::defaultIterationCountPerKernel( iterationCount);
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID Vector<int, 1> pointCoordinates(const TView3 &view, const Vector<int, 1>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = blockSize();
		return detail::defaultPointCoordinates( base, index, max_grid_size, block_size);
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID Vector<int, 2> pointCoordinates(const TView3 &view, const Vector<int, 2>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = blockSize();
		const dim3 iterationsCount = iterationCounts(view);
		return detail:: defaultPointCoordinates(base, index, max_grid_size, block_size, iterationsCount);
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID Vector<int, 3> pointCoordinates(const TView3 &view, const Vector<int, 3>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = blockSize();
		const dim3 iterationsCount = iterationCounts(view);
		return detail:: defaultPointCoordinates(base, index, max_grid_size, block_size, iterationsCount);
	}

	template<typename TView3>
	static BOLT_DECL_HYBRID dim3 gridCount(const TView3 &view)
	{
		const dim3 grid_count = gridSize(view);
		const dim3 max_size = maxGridSize();
		return detail::defaultKernelThreadGeometry(grid_count, max_size);
	}
#endif  //defined(__CUDACC__)
};

namespace detail {
	template <typename TFunctor, typename TView, typename TOutView,typename TPolicy>
	struct TransformFunctor
	{
		TFunctor functor_;
		TOutView out_view_;
		TransformFunctor(TFunctor functor, TOutView out_view):
			functor_(functor),
			out_view_(out_view){};
		BOLT_HD_WARNING_DISABLE
		BOLT_DECL_HYBRID void operator() ( const detail::ViewIndexingLocator<TView>& point, const Vector<int, TView::kDimension> & coord) const
		{
			out_view_[coord] = functor_(point.get());
		}
	};
} // namespace detail

/** \ingroup Algorithms
 * @{
 **/

/// Applies an operation on each element of the input view and stores the result in the output view.
/// Both views must have same size
/// Input and output view can the same if the functor does not have side effects changing the data in passed views.
/// \param in_view Input view, can be read only.
/// \param out_view Output view - must provide write access.
/// \param functor Applied callable object
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transform(TInView in_view, TOutView out_view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	static_assert(TInView::kIsDeviceView && TOutView::kIsDeviceView || TInView::kIsHostView && TOutView::kIsHostView, "For transform both views should be usabele either on device or on host. You can not transform between device and host view.");
	static_assert(TInView::kDimension == TOutView::kDimension, "Both input and output should have the same dimension");
	BOLT_ASSERT(in_view.size() == out_view.size());
	if (in_view.size() != out_view.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(in_view.size(), out_view.size()));
	}


	if (isEmpty(in_view)) {
		return;
	}
	detail::TransformFunctor<TFunctor, TInView, TOutView, TPolicy> lambda (functor, out_view);
	detail::IterateImplementation<TInView::kIsDeviceView && TOutView::kIsDeviceView, detail::ViewIndexingLocator<TInView>>::run(in_view, lambda, policy, cuda_stream);
}


/// Applies an operation on each element of the input view and stores the result in the output view.
/// Both views must have same size
/// Input and output view can the same if the functor does not have side effects changing the data in passed views.
/// \param in_view Input view, can be read only.
/// \param functor Applied callable object
/// \param out_view Output view - must provide write access.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TInView, typename TOutView, typename TFunctor>
void
transform(TInView in_view, TOutView out_view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	transform(in_view, out_view, functor, DefaultTransformPolicy<TInView, TOutView>(), cuda_stream);
}

/**
 * @}
 **/

//*************************************************************************************************************

namespace detail {
	template <typename TFunctor, typename TView, typename TOutView,typename TPolicy>
	struct TransformPositionFunctor
	{
		TFunctor functor_;
		TOutView out_view_;
		TransformPositionFunctor(TFunctor functor, TOutView out_view):
			functor_(functor),
			out_view_(out_view){};

		BOLT_HD_WARNING_DISABLE
		template<typename TCoordinates>
		BOLT_DECL_HYBRID
		void operator() (const detail::ViewIndexingLocator<TView> & point, const TCoordinates & coord) const
		{
			// out_view_[coord] = functor_(point.get(), coord);
			out_view_[coord] = functor_(point.get(), point.location_);
		}
	};
} // namespace detail

/** \ingroup Algorithms
 * @{
 **/

/// Applies an operation on each element of the input view and stores the result in the output view, Functor obtains also element index as an argument together with element value.
/// Both views must have same size
/// Input and output view can the same if the functor does not have side effects changing the data in passed views.
/// \param in_view Input view, can be read only.
/// \param out_view Output view - must provide write access.
/// \param functor Applied callable object
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transformPosition(TInView in_view, TOutView out_view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	static_assert(TInView::kIsDeviceView && TOutView::kIsDeviceView || TInView::kIsHostView && TOutView::kIsHostView, "For transform both views should be usabele either on device or on host. You can not transform between device and host view.");
	static_assert(TInView::kDimension == TOutView::kDimension, "Both input and output should have the same dimension");
	BOLT_ASSERT(in_view.size() >= out_view.size());
	if (in_view.size() != out_view.size()) {
		BOLT_THROW(IncompatibleViewSizes() << getViewPairSizesErrorInfo(in_view.size(), out_view.size()));
	}

	detail::TransformPositionFunctor<TFunctor, TInView, TOutView, TPolicy> lambda (functor, out_view);
	if (isEmpty(in_view)) {
		return;
	}

	detail::IterateImplementation<TInView::kIsDeviceView && TOutView::kIsDeviceView, detail::ViewIndexingLocator<TInView>>::run(in_view, lambda, policy, cuda_stream);
}

/// Applies an operation on each element of the input view and stores the result in the output view, Functor obtains also element index as an argument together with element value.
/// Both views must have same size
/// Input and output view can the same if the functor does not have side effects changing the data in passed views.
/// \param in_view Input view, can be read only.
/// \param out_view Output view - must provide write access.
/// \param functor Applied callable object
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TInView, typename TOutView, typename TFunctor>
void
transformPosition(TInView in_view, TOutView out_view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	transformPosition(in_view, out_view, functor, DefaultTransformPolicy<TInView, TOutView>(), cuda_stream);
}

/**
 * @}
 **/


template<typename TView1, typename TView2, BorderHandling tBorderHandling = BorderHandling::kRepeat, bool tPreloadToSharedMemory = false>
struct DefaultTransformLocatorPolicy : DefaultTransformPolicy<TView1, TView2> {
	static constexpr BorderHandling kBorderHandling = tBorderHandling;
	static constexpr bool kPreloadToSharedMemory = tPreloadToSharedMemory;
	using SizeType = typename TView1::SizeType;

	void setPreload(SizeType size, SizeType center){
		overlapStart = center;
		overlapEnd = size - center - SizeType::fill(1);
	}

	SizeType overlapStart;
	SizeType overlapEnd;

};

// template<typename TView1, typename TView2, BorderHandling tBorderHandling = BorderHandling::kRepeat, bool tPreloadToSharedMemory = true>
// struct DefaultConvolutionPolicy : DefaultTransformLocatorPolicy<TView1, TView2, tBorderHandling, tPreloadToSharedMemory> {
//
// };
template<typename TView1, typename TView2, BorderHandling tBorderHandling = BorderHandling::kRepeat, bool tPreloadToSharedMemory = true>
DefaultTransformLocatorPolicy<TView1, TView2, tBorderHandling, tPreloadToSharedMemory>
getDefaultConvolutionPolicy(TView1  /*view1*/, TView2  /*view2*/){
	return DefaultTransformLocatorPolicy<TView1, TView2, tBorderHandling, tPreloadToSharedMemory>();
}

namespace detail {
	template <typename TFunctor, typename TView, typename TOutView, typename TPolicy>
	struct TransformLocatorFunctor
	{
		TFunctor functor_;
		TOutView out_view_;
		TransformLocatorFunctor(TFunctor functor, TOutView out_view):
			functor_(functor),
			out_view_(out_view){};

		BOLT_HD_WARNING_DISABLE
		template<typename TLocator, typename TCoordinates>
		BOLT_DECL_HYBRID void operator() (const TLocator & locator, const TCoordinates &coord) const
		{
			out_view_[coord] = functor_(locator);
		}
	};

	template <typename TFunctor, typename TView, typename TOutView,typename TPolicy>
	struct TransformLocatorPositionFunctor
	{
		TFunctor functor_;
		TOutView out_view_;
		TransformLocatorPositionFunctor(TFunctor functor, TOutView out_view):
			functor_(functor),
			out_view_(out_view){};
		BOLT_HD_WARNING_DISABLE
		template<typename TLocator> BOLT_DECL_HYBRID void operator() (const TLocator & locator, const Vector<int, TView::kDimension> & coord) const
		{
			out_view_[coord] = functor_(locator);
		}
	};
} // namespace detail

/** \ingroup Algorithms
 * @{
 **/

/// Applies an operation for each each element of the output view.
/// The operator is supplied an image locator, so the relative element neighborhood is available, thus allowing implementation of filters like convolution.
/// In this case the views cannot map the same memory block, because it can introduce race conditions.
/// \param in_view Input view, can be read only.
/// \param out_view Output view - must provide write access.
/// \param functor Applied callable object
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Which stream should schedule this operation
template <typename TInView, typename TOutView, typename TFunctor, typename TPolicy>
void
transformLocator(TInView in_view, TOutView out_view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	static_assert(TInView::kIsDeviceView && TOutView::kIsDeviceView || TInView::kIsHostView && TOutView::kIsHostView, "For transform both views should be usabele either on device or on host. You can not transform between device and host view.");
	static_assert(TInView::kDimension == TOutView::kDimension, "Both input and output should have the same dimension");
	if (isEmpty(in_view)) {
		return;
	}

	detail::TransformLocatorFunctor<TFunctor, TInView, TOutView, TPolicy> lambda (functor, out_view);
	detail::IterateImplementation<TInView::kIsDeviceView && TOutView::kIsDeviceView, detail::LocatorConstructor<TInView, TPolicy::kPreloadToSharedMemory> >::run(in_view, lambda, policy, cuda_stream);
}

/// Applies an operation for each each element of the output view.
/// The operator is supplied an image locator, so the relative element neighborhood is available, thus allowing implementation of filters like convolution.
/// In this case the views cannot map the same memory block, because it can introduce race conditions.
/// \param in_view Input view, can be read only.
/// \param out_view Output view - must provide write access.
/// \param functor Applied callable object
/// \param cuda_stream Which stream should schedule this operation
template <typename TInView, typename TOutView, typename TFunctor>
void
transformLocator(TInView in_view, TOutView out_view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	transformLocator(in_view, out_view, functor, DefaultTransformLocatorPolicy<TInView, TOutView>(), cuda_stream);
}

/**
 * @}
 **/

//*************************************************************************************************************
}//namespace bolt
