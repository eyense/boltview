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
	int j = from[1];
	for(; j < to[1]; ++j) {
		int i = from[0];
		for(; i < to[0] - 3; i+=4) {
			sum += callable({i,j});
			sum += callable({i+1,j});
			sum += callable({i+2,j});
			sum += callable({i+3,j});
		}
		for(; i < to[0]; ++i) {
			sum += callable({i,j});
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
	int k = from[2];
	for(; k < to[2]; ++k) {
		int j = from[1];
		for(; j < to[1]; ++j) {
			int i = from[0];
			for(; i < to[0] - 3; i+=4) {
				sum += callable({i, j, k});
				sum += callable({i+1, j, k});
				sum += callable({i+2, j, k});
				sum += callable({i+3, j, k});
			}
			for(; i < to[0]; ++i) {
				sum += callable({i, j, k});
			}
		}
	}
	return sum;
}


}  // namespace bolt
