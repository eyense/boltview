#pragma once

#include <boltview/detail/algorithm_common.h>
#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/image_locator.h>
#include <boltview/detail/meta_algorithm_utils.h>

namespace bolt {

template<typename TView>
struct DefaultForEachPolicyBase {
#if defined(__CUDACC__)
	static BOLT_DECL_HYBRID dim3 blockSize()
	{
		return detail::defaultBlockDimForDimension<TView::kDimension>();
	}

	static BOLT_DECL_HYBRID dim3 gridSize(const TView &view)
	{
        	return detail::defaultGridSizeForBlockDim(dataSize(view), blockSize());
	}
#endif  // __CUDACC__

};

template<typename TView, typename TBase>
struct DefaultForEachPolicyMixin: TBase {
#if defined(__CUDACC__)
	using Base = TBase;

	static BOLT_DECL_HYBRID dim3 maxGridSize()
	{
		return detail::defaultMaxGridSize();
	}

	static BOLT_DECL_HYBRID dim3 gridCount(const TView &view)
	{
		const dim3 grid_count = Base::gridSize(view);
		const dim3 max_size = maxGridSize();
		return  detail::defaultKernelThreadGeometry(grid_count, max_size);
	}

	static BOLT_DECL_HYBRID dim3 iterationCounts(const TView &view)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 grid_count =  Base::gridSize(view);
		return detail::defaultIterationCountsPerKernel( grid_count, max_grid_size);
	}

	static BOLT_DECL_HYBRID int pointCountPerKernel(const TView &view)
	{
		const dim3 iteration_count = iterationCounts(view);
		return detail::defaultIterationCountPerKernel(iteration_count);
	}

	static BOLT_DECL_HYBRID Vector<int, 1> pointCoordinates(const TView &view, const Vector<int, 1>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = Base::blockSize();
		return  detail::defaultPointCoordinates( base, index, max_grid_size, block_size);
	}

	static BOLT_DECL_HYBRID Vector<int, 2> pointCoordinates(const TView &view, const Vector<int, 2>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = Base::blockSize();
		const dim3 iteration_counts = iterationCounts(view);
		return detail:: defaultPointCoordinates(base, index, max_grid_size, block_size, iteration_counts);
	}

	static BOLT_DECL_HYBRID Vector<int, 3> pointCoordinates(const TView &view, const Vector<int, 3>& base, int index)
	{
		const dim3 max_grid_size = maxGridSize();
		const dim3 block_size = Base::blockSize();
		const dim3 iterationsCount = iterationCounts(view);
		return detail:: defaultPointCoordinates(base, index, max_grid_size, block_size, iterationsCount);
	}

#endif  // __CUDACC__

};

template<typename TView>
using DefaultForEachPolicy = DefaultForEachPolicyMixin<TView, DefaultForEachPolicyBase<TView>>;

namespace detail {
	template <typename TFunctor, typename TView, typename TPolicy>
	struct ForEachFunctor
	{
		TFunctor functor_;
		explicit ForEachFunctor(TFunctor functor):
		   functor_(functor){};
		BOLT_HD_WARNING_DISABLE
		BOLT_DECL_HYBRID void operator() (const detail::ViewIndexingLocator<TView>& point, const typename TView::IndexType &  /*coord*/) const
		{
			static_assert(std::is_void<decltype(functor_(point.get()))>::value, "Return value of the functor is ignored in ForEach loop");
			functor_(point.get());
		}
	};
} // namespace detail


/** \ingroup Algorithms
 * @{
 **/

/// Apply functor on each element in passed view
/// \param view Processed image view
/// \param functor functor which is called on each element. Whether it get reference or const reference for the processed element depends on type of the view.
/// \param policy You can specify kernel execution configuration.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TView, typename TFunctor, typename TPolicy>
void
forEach(TView view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	if (isEmpty(view)) {
		return;
	}
	detail::ForEachFunctor<TFunctor, TView, TPolicy> lambda{functor};
	detail::IterateImplementation<TView::kIsDeviceView, detail::ViewIndexingLocator<TView>>::run(view, lambda, policy, cuda_stream);
}

/// Apply functor on each element in passed view
/// \param view Processed image view
/// \param functor functor which is called on each element. Whether it get reference or const reference for the processed element depends on type of the view.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TView, typename TFunctor>
void
forEach(TView view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	forEach(view, functor, DefaultForEachPolicy<TView>(), cuda_stream);
}

/**
 * @}
 **/



//*************************************************************************************************************

namespace detail {
	template <typename TFunctor, typename TView, typename TPolicy>
	struct ForEachPositionFunctor
	{
		TFunctor functor_;
		explicit ForEachPositionFunctor(TFunctor functor):
		   functor_(functor){ };
		BOLT_HD_WARNING_DISABLE
		BOLT_DECL_HYBRID void operator() (const detail::ViewIndexingLocator<TView> & point, const Vector<int, TView::kDimension> & coord) const
		{
			static_assert(std::is_void<decltype(functor_(point.get(), coord ) )>::value, "Return value of the functor is ignored in ForEach loop");
			functor_(point.get(), coord);
		}
	};
} // namespace detail

/** \ingroup Algorithms
 * @{
 **/

/// Apply functor on each element in passed view together with the element's coordinate.
/// \param view Processed image view
/// \param functor functor which is called on each element and its index. Whether it get reference or const reference for the processed element depends on type of the view.
/// \param policy You can specify kernel execution configuration.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TView, typename TFunctor, typename TPolicy>
void
forEachPosition(TView view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	if (isEmpty(view)) {
		return;
	}
	detail::ForEachPositionFunctor<TFunctor, TView, TPolicy> lambda (functor);
	detail::IterateImplementation<TView::kIsDeviceView, detail::ViewIndexingLocator<TView>>::run(view, lambda, policy, cuda_stream);
}

/// Apply functor on each element in passed view together with the element's coordinate.
/// \param view Processed image view
/// \param functor functor which is called on each element and its index. Whether it get reference or const reference for the processed element depends on type of the view.
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TView, typename TFunctor>
void
forEachPosition(TView view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	forEachPosition(view, functor, DefaultForEachPolicy<TView>(), cuda_stream);
}

/**
 * @}
 **/

namespace detail {
	template <typename TFunctor, typename TView, typename TPolicy>
	struct ForEachLocatorFunctor
	{
		TFunctor functor_;
		explicit ForEachLocatorFunctor(TFunctor functor):
			functor_(functor){};

		BOLT_HD_WARNING_DISABLE
		template<typename TLocator>
		BOLT_DECL_HYBRID void operator() (const TLocator & locator, const Vector<int, TView::kDimension> & coord) const
		{
			static_assert(std::is_void<decltype(functor_(locator, coord))>::value, "Return value of the functor is ignored in ForEach loop");
			//The functor almost surely needs the original coordinate - the coordinate in the locator may be shifted...
			//If the locator is constructed from the shared memory, the Coord() method may return shifted coordinates.
			// Also it may return wrong coordinates on big files
			functor_(locator, coord);
		}
	};
} // namespace detail

template<typename TView1, BorderHandling tBorderHandling = BorderHandling::kRepeat, bool tPreloadToSharedMemory = false>
struct DefaultForEachLocatorPolicy : DefaultForEachPolicy<TView1> {
	static constexpr BorderHandling kBorderHandling = tBorderHandling;
	static constexpr bool kPreloadToSharedMemory = tPreloadToSharedMemory;

	void setPreload(Vector<int, TView1::kDimension> size, Vector<int, TView1::kDimension> center){
		overlapStart = center;
		overlapEnd = size - center - Vector<int, TView1::kDimension>::Fill(1);
	}

	Vector<int, TView1::kDimension> overlapStart;
	Vector<int, TView1::kDimension> overlapEnd;

};

/** \ingroup Algorithms
 * @{
 **/

/// Applies an operation for each each element of the view.
/// The operator is supplied an image locator, so the relative element neighborhood is available, thus allowing implementation of filters like convolution.
/// In this case the views cannot map the same memory block, because it can introduce race conditions.
/// \param view  Processed image view.
/// \param functor Applied callable object
/// \param policy Policy class describing kernel execution configuration.
/// \param cuda_stream Which stream should schedule this operation
template <typename TView, typename TFunctor, typename TPolicy>
void
forEachLocator(TView view, TFunctor functor, TPolicy policy, cudaStream_t cuda_stream = nullptr)
{
	if (isEmpty(view)) {
		return;
	}
        static_assert(!std::is_reference<typename TView::AccessType>::value || std::is_const<typename TView::AccessType>::value,
		"ForeachLocator should be used on constant view. If you try modifying the view you will get race conditions.");
	detail::ForEachLocatorFunctor<TFunctor, TView, TPolicy> lambda{functor};
	detail::IterateImplementation<TView::kIsDeviceView, detail::LocatorConstructor<TView, TPolicy::kPreloadToSharedMemory> >::run(view, lambda, policy, cuda_stream);
}

/// Applies an operation for each each element of the view.
/// The operator is supplied an image locator, so the relative element neighborhood is available.
/// \param view  Processed image view.
/// \param functor Applied callable object
/// \param cuda_stream Which stream should schedule this operation
template <typename TView, typename TFunctor>
void
forEachLocator(TView view, TFunctor functor, cudaStream_t cuda_stream = nullptr)
{
	forEachLocator(view, functor, DefaultForEachLocatorPolicy<TView>(), cuda_stream);
}

/**
 * @}
 **/


}//namespace bolt
