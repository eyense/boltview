#pragma once

#include <boltview/image_locator.h>

// TODO
namespace bolt {

namespace detail {

template <typename TInView, typename TOutView, typename TFunctor>
CUGIP_GLOBAL void
kernel_filter(TInView in_view, TOutView out_view, TFunctor aOperator )
{
	typename TOutView::coord_t coord(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	typename TOutView::extents_t extents = out_view.dimensions();

	if (coord < extents) {
		out_view[coord] = aOperator(in_view.template locator<cugip::border_handling_repeat_t>(coord));
	}
}

}//namespace detail


/** \ingroup Algorithms
 * @{
 **/

/// For each element from input view apply spatial filter with bounded support (limited neighborhood) - convolution, median filter, morphological operations, etc.
/// Size of filter mask should be known in compile time, so the execution can be internaly optimized by preloading to shared memory. Too large neighborhood causes fallback to TransformLocator.
/// Input view must be same or bigger then the output view. Accessed memory for input and output view must be different otherwise it can introduce race conditions.
/// \param in_view Input view - can be read only
/// \param out_view Output view - needs write access to its elements
/// \param filter_operator Filter operator - it accesses element neighborhood by using image locators
/// \param policy Policy describing kernel execution configuration and boundary problem handling configuration in image locators.
/// \sa TransformLocator
template <typename TInView, typename TOutView, typename TFilterOperator, typename TPolicy>
void
filter(TInView in_view, TOutView out_view, TFilterOperator filter_operator, TPolicy policy)
{
	dim3 blockSize(256, 1, 1);
	dim3 gridSize((in_view.dimensions().template get<0>() / blockSize.x + 1), in_view.dimensions().template get<1>() / blockSize.y + 1, 1);

	D_PRINT("Executing kernel: blockSize = "
	               << blockSize
	               << "; gridSize = "
	               << gridSize
	       );
	detail::kernel_filter<TInView, TOutView, TFunctor>
		<<<gridSize, blockSize>>>(in_view, out_view, aOperator);
	CUGIP_CHECK_ERROR_STATE("kernel_for_each");
}

/// For each element from input view apply spatial filter with bounded support (limited neighborhood) - convolution, median filter, morphological operations, etc.
/// Size of filter mask should be known in compile time, so the execution can be internaly optimized by preloading to shared memory. Too large neighborhood causes fallback to TransformLocator.
/// Input view must be same or bigger then the output view. Accessed memory for input and output view must be different otherwise it can introduce race conditions.
/// \param in_view Input view - can be read only
/// \param out_view Output view - needs write access to its elements
/// \param filter_operator Filter operator - it accesses element neighborhood by using image locators
/// \sa TransformLocator
template <typename TInView, typename TOutView, typename TFilterOperator>
void
filter(TInView in_view, TOutView out_view, TFilterOperator filter_operator)
{
	Filter(in_view, out_view, filter_operator, DefaultFilterPolicy());
}

/**
 * @}
 **/

//*************************************************************************************************************

}//namespace bolt

