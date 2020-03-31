// Copyright 2015 Eyen SE
// Author: Pavel Mikus pavel.mikus@eyen.se
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include "cuda_defines.h"
#include "math/vector.h"

#include <boltview/image_locator.h>
#include <boltview/interpolated_view.h>

namespace bolt
{
template<typename TType>
BOLT_DECL_HYBRID auto lerp(float weight, TType value1, TType value2) -> decltype(weight * value1)
{
	return (1.0f - weight) * value1 + weight * value2;
	// More optimal floating point implementation:
	// return fmaf(weight, value2, fmaf(-weight, value1, value1));
	// TODO(johny) - use in float overload.
}


/// Do nearest neighbor interpolation of image padded by specified background value.
template<typename TBoundaryHandler>
struct NearestNeighborInterpolator {
	// NearestNeighborInterpolator() = default;
	// explicit NearestNeighborInterpolator(TBoundaryHandler &&boundary_handler):
	// boundary_handler_(std::forward(boundary_handler))
	// {}
	template<typename TView>
	BOLT_DECL_HYBRID static typename TView::Element compute(const TView &view, const Vector<float, TView::kDimension> &position)
	{
		// if fill(0.5f) is used, access on position Float*(0.0f,0.0f,..) rounds to (-1,-1,...)
		// TODO(johny) - constexpr
		const auto k_half_offset = Vector<float, TView::kDimension>::fill(0.499999f);
		auto index = Vector<int, TView::kDimension>(round(position - k_half_offset));
		return TBoundaryHandler::access(view, index, Vector<int, TView::kDimension>::fill(0));
	}
	// TBoundaryHandler boundary_handler_;

  	  	template<typename TView>
	BOLT_DECL_HYBRID static typename TView::Element boundedAccess(const TView &view, const  Vector<int, TView::kDimension> &index)
	{
		return TBoundaryHandler::access(view, index, Vector<int, TView::kDimension>::fill(0));
	}

};

template<typename TBoundaryHandler, int tTDimension>
struct LinearInterpolatorImpl {
	template<typename TView>
	BOLT_DECL_HYBRID static auto compute(const TView &view, const Vector<float, TView::kDimension> &position)
		-> decltype(0.0f * view[Vector<typename TView::IndexType, TView::kDimension>::fill(0)])
	{
		auto corner1 = position;
		corner1[tTDimension - 1] = floor(corner1[tTDimension - 1]);
		auto corner2 = position;
		corner2[tTDimension - 1] = ceil(corner2[tTDimension - 1]);
		float weight = position[tTDimension - 1] - floor(position[tTDimension - 1]);
		auto value1 = LinearInterpolatorImpl<TBoundaryHandler, tTDimension - 1>::compute(view, corner1);
		auto value2 = LinearInterpolatorImpl<TBoundaryHandler, tTDimension - 1>::compute(view, corner2);
		return lerp(weight, value1, value2);
	}
};

template<typename TBoundaryHandler>
struct LinearInterpolatorImpl<TBoundaryHandler, 1> {
	template<typename TView>
	BOLT_DECL_HYBRID static auto compute(const TView &view, const Vector<float, TView::kDimension> &position)
		-> decltype(0.0f * view[Vector<typename TView::IndexType, TView::kDimension>::fill(0)])
	{
		auto corner1 = position;
		corner1[0] = floor(corner1[0]);
		auto corner2 = position;
		corner2[0] = ceil(corner2[0]);
		float weight = position[0] - floor(position[0]);
		auto index1 = Vector<typename TView::TIndex, TView::kDimension>(corner1);
		auto index2 = Vector<typename TView::TIndex, TView::kDimension>(corner2);
		auto value1 = TBoundaryHandler::access(view, index1, Vector<typename TView::TIndex, TView::kDimension>::fill(0));
		auto value2 = TBoundaryHandler::access(view, index2, Vector<typename TView::TIndex, TView::kDimension>::fill(0));
		return lerp(weight, value1, value2);
	}
};

/// Do linear interpolation of image
BOLT_HD_WARNING_DISABLE
template<typename TBoundaryHandler>
struct LinearInterpolator {
	template<typename TView>
	BOLT_DECL_HYBRID static typename TView::Element compute(const TView &view, const Vector<float, TView::kDimension> &position)
	{
		auto result = LinearInterpolatorImpl<TBoundaryHandler, TView::kDimension>::compute(view, position);
		return static_cast<typename TView::Element>(result);
	}

  	template<typename TView>
  	BOLT_DECL_HYBRID static typename TView::Element boundedAccess(const TView &view, const  Vector<int, TView::kDimension> &index)
  	{
  	  	return TBoundaryHandler::access(view, index, Vector<int, TView::kDimension>::fill(0));
  	}
};


struct CatmullRomWeights {
	BOLT_DECL_HYBRID
	static Vector<float, 4> compute(float offset)
	{
		Vector<float, 4> result;
		result[0] = compute0(offset);
		result[1] = compute1(offset);
		result[2] = compute2(offset);
		result[3] = compute3(offset);
		return result;
	}
	// TODO(johny) - check correctnes and use Horner's scheme
	BOLT_DECL_HYBRID
	static float compute0(float offset)
	{
		// -o^3t + 2o^2t - ot
		return ((-offset + 2.0f) * offset - 1.0f) * offset * kTension;
	}

	BOLT_DECL_HYBRID
	static float compute1(float offset)
	{
		return offset * offset * offset * (2.0f - kTension) + offset * offset * (kTension - 3.0f) + 1.0f;
	}

	BOLT_DECL_HYBRID
	static float compute2(float offset)
	{
		return offset * offset * offset * (kTension - 2.0f) + offset * offset * (3.0f - 2.0f * kTension) + offset * kTension;
	}

	BOLT_DECL_HYBRID
	static float compute3(float offset)
	{
		return offset * offset * offset * kTension - offset * offset * kTension;
	}

	static constexpr float kTension = 0.5f;
};


template<typename TBoundaryHandler, int tTDimension>
struct CubicInterpolatorImpl {
	template<typename TView>
	BOLT_DECL_HYBRID static auto compute(const TView &view, const Vector<float, TView::kDimension> &position)
		-> decltype(0.0f * view[Vector<typename TView::IndexType, TView::kDimension>::fill(0)])
	{
		float weight = position[tTDimension - 1] - floor(position[tTDimension - 1]);
		auto weights = CatmullRomWeights::compute(weight);
		float g0 = weights[0] + weights[1];
		float g1 = weights[2] + weights[3];
		float h0 = weights[1] / g0;
		float h1 = g1 != 0 ? weights[3] / g1 : 0.0f;
		auto sample_position1 = position;
		sample_position1[tTDimension - 1] = floor(position[tTDimension - 1]) - 1 + h0;
		auto sample_position2 = position;
		sample_position2[tTDimension - 1] = ceil(position[tTDimension - 1]) + h1;

		return g0 * CubicInterpolatorImpl<TBoundaryHandler, tTDimension - 1>::compute(view, sample_position1) +
			   g1 * CubicInterpolatorImpl<TBoundaryHandler, tTDimension - 1>::compute(view, sample_position2);
	}
};

template<typename TBoundaryHandler>
struct CubicInterpolatorImpl<TBoundaryHandler, 1> {
	template<typename TView>
	BOLT_DECL_HYBRID static auto compute(const TView &view, const Vector<float, TView::kDimension> &position)
		-> decltype(0.0f * view[Vector<typename TView::IndexType, TView::kDimension>::fill(0)])
	{
		float weight = position[0] - floor(position[0]);
		auto weights = CatmullRomWeights::compute(weight);
		float g0 = weights[0] + weights[1];
		float g1 = weights[2] + weights[3];
		float h0 = weights[1] / g0;
		float h1 = g1 != 0 ? weights[3] / g1 : 0.0f;
		auto sample_position1 = position;
		sample_position1[0] = floor(position[0]) - 1 + h0;
		auto sample_position2 = position;
		sample_position2[0] = ceil(position[0]) + h1;

		return g0 * LinearInterpolator<TBoundaryHandler>::compute(view, sample_position1) +
			   g1 * LinearInterpolator<TBoundaryHandler>::compute(view, sample_position2);
	}
};


template<typename TBoundaryHandler>
struct CubicInterpolator {
	template<typename TView>
	BOLT_DECL_HYBRID static typename TView::Element compute(const TView &view, const Vector<float, TView::kDimension> &position)
	{
		auto result = CubicInterpolatorImpl<TBoundaryHandler, TView::kDimension>::compute(view, position);
		return static_cast<typename TView::Element>(result);
	}

  	  	template<typename TView>
  	  	BOLT_DECL_HYBRID static typename TView::Element boundedAccess(const TView &view, const  Vector<int, TView::kDimension> &index)
  	  	{
  	  	  	  	return TBoundaryHandler::access(view, index, Vector<int, TView::kDimension>::fill(0));
  	  	}
};

using NearestRepeat = NearestNeighborInterpolator<BorderHandlingTraits<BorderHandling::kRepeat>>;
using LinearRepeat = LinearInterpolator<BorderHandlingTraits<BorderHandling::kRepeat>>;
using CubicRepeat = CubicInterpolator<BorderHandlingTraits<BorderHandling::kRepeat>>;


};  // namespace bolt
