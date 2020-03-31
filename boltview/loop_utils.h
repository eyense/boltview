// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#pragma once

#include <boltview/math/vector.h>

namespace bolt {


/// Sums results of callable applied for each index in <from, to> interval
/// \param from one corner of the 2D interval (is part of the processed range)
/// \param to second coner of the 2D interval (is NOT part of the processed range)
/// \param init_value Value used for inicialization of the sum
/// \param callable Callable taking one argument in form of 2D index and returning value for summation.
BOLT_HD_WARNING_DISABLE
template<typename TValue, typename TCallable>
BOLT_DECL_HYBRID
TValue sumEachNeighbor(Int2 from, Int2 to, TValue init_value, TCallable callable)
{
	TValue sum = init_value;
	Int2 index;
	for(index[1] = from[1]; index[1] < to[1]; ++index[1]) {
		for(index[0] = from[0]; index[0] < to[0]; ++index[0]) {
			sum += callable(index);
		}
	}
	return sum;
}

/// Sums results of callable applied for each index in <from, to> interval
/// \param from one corner of the 3D interval (is part of the processed range)
/// \param to second coner of the 3D interval (is NOT part of the processed range)
/// \param init_value Value used for inicialization of the sum
/// \param callable Callable taking one argument in form of 3D index and returning value for summation.
BOLT_HD_WARNING_DISABLE
template<typename TValue, typename TCallable>
BOLT_DECL_HYBRID
TValue sumEachNeighbor(Int3 from, Int3 to, TValue init_value, TCallable callable)
{
	TValue sum = init_value;
	Int3 index;
	for(index[2] = from[2]; index[2] < to[2]; ++index[2]) {
		for(index[1] = from[1]; index[1] < to[1]; ++index[1]) {
			for(index[0] = from[0]; index[0] < to[0]; ++index[0]) {
				sum += callable(index);
			}
		}
	}
	return sum;
}


}  // namespace bolt
