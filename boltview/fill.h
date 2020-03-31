#pragma once

#include <boltview/for_each.h>

namespace bolt {

namespace detail {

template<typename TValue>
struct AssignFunctor {

	template<typename TOutput>
	BOLT_DECL_HYBRID
	void operator()(TOutput &output) const {
		output = value;
	}
	TValue value;
};

template<typename TValue>
AssignFunctor<TValue> createAssignFunctor(TValue value) {
	return AssignFunctor<TValue>{ value };
}

}  //namespace detail

/** \ingroup Algorithms
 * @{
 **/

/// Assigns the same value to each element in passed view
/// \param view Processed image view
/// \param value Value assigned to all image elements
/// \param cuda_stream Id of the cuda stream on which the code will be executed (valid only for device/hybrid views)
template <typename TView, typename TValue>
void fill(TView view, TValue value, cudaStream_t cuda_stream = nullptr)
{
	forEach(view, detail::createAssignFunctor(value), cuda_stream);
}

// TODO(johny) version with policy


/**
 * @}
 **/


}  //namespace bolt
