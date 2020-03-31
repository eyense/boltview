// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once


#if defined(__CUDACC__)
	#include <thrust/device_vector.h>
	#include <thrust/reduce.h>
	#include <boltview/cuda_utils.h>
	#include <boltview/device_image.h>
#endif  // __CUDACC__

#include <boltview/image_view_utils.h>
#include <boltview/execution_utils.h>

namespace bolt {

/// \addtogroup Algorithms
/// @{

/// Implements parallel reduction - application of associative operator on all image elements.
/// For example to sum all values in integer image:
/// \code
/// 	sum = reduce(view, 0, thrust::plus<int>());
/// \endcode
/// \param view Processed image view - only constant element access needed.
/// \param initial_value Value used for result initialization (0 for sums, 1 for products, etc.)
/// \param reduction_operator Associative operator - it must be callable on device like this: result = reduction_operator(val1, val2).
/// \param execution_policy Policy which can influence the execution (thread counts, tec.)
/// \return Result of the operation.
template<typename TView, typename TOutputValue, typename TOperator>
TOutputValue reduce(TView view, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy = ExecutionPolicy{});


template<typename TView, typename TOutputView, typename TOutputValue, int tDimension, typename TOperator>
void dimensionReduce(TView view, TOutputView output_view, DimensionValue<tDimension> dimension, TOutputValue initial_value, TOperator reduction_operator, ExecutionPolicy execution_policy = ExecutionPolicy{});

template<typename TView, typename TOutputValue, class = typename std::enable_if<IsImageView<TView>::value>::type>
TOutputValue sum(TView view, TOutputValue initial_value, ExecutionPolicy execution_policy = ExecutionPolicy{});

/// WARNING: as a result type is used element type of the passed view, so be aware of possible overflow if summing large view of small type elements (int8_t, int16_t, ...)
/// When in doubt use sum(view, initial_value); and pass initial value of bigger type to prevent overflows.
template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
typename TView::Element sum(TView view, ExecutionPolicy execution_policy = ExecutionPolicy{});

template<typename TView1, typename TView2, typename TOutputValue>
TOutputValue sumSquareDifferences(TView1 view1, TView2 view2, TOutputValue initial_value, ExecutionPolicy execution_policy = ExecutionPolicy{});

template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
bool isFinite(TView view, ExecutionPolicy execution_policy = ExecutionPolicy{});

/// @}

}  // namespace bolt

#include "reduce.tcc"
