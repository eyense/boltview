// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <boltview/for_each.h>
#include <boltview/math/quaternion.h>
#include <boltview/device_image_view_base.h>
#include <boltview/interpolation.h>
#include <boltview/interpolated_view.h>

namespace bolt {


/*template<typename TView, int tDim, typename std::enable_if<is_interpolated_view<TView>::value, int>::type = 0>
CUGIP_DECL_HYBRID simple_vector<float, tDim>
coordinatesFromIndex(TView aView, simple_vector<int, tDim> aIndex)
{
	return aView.coordinates_from_index(aIndex);
}*/

template<typename TView, int tDim>
BOLT_DECL_HYBRID Vector<float, tDim>
coordinatesFromIndex(TView aView, Vector<int, tDim> aIndex)
{
	return aIndex + Vector<float, tDim>::fill(0.5f);
}


/// Provides transformed access to the provide image view
/// \tparam TView type of transformed image view
/// \tparam TInverseOperator operator which provides access to original image from the transformed image,
/// so it should implement inverse transformation. Called as inv(view, index, background_value), where view is the original view,
/// index is element index in the transformed image view and background_value is used for elements leading outside from original image view.
template<typename TView, typename TInverseOperator>
class TransformedImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const int kDimension = TView::kDimension;
	static const bool kIsMemoryBased = false;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Element = typename TView::Element;
	using AccessType = Element;

	TransformedImageView(TView view, const SizeType &size, const TInverseOperator &inverse_operator) :
		Predecessor(size),
		view_(view),
		inverse_operator_(inverse_operator)
	{
		// D_FORMAT("Transformed view:\n\tSize: %1%\n\tAccessed size: %2%", size, view.size());
	}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		return inverse_operator_(view_, index);
	}

protected:
	TView view_;
	TInverseOperator inverse_operator_;
};

/// Rotated access operator with quaternion describing axis and angle with coordinates of the center of the rotation.
template<typename TInterpolator>
struct RotatedAccessOperator {
	RotatedAccessOperator(const Quaternion<float> &rotation_quaternion, const Float3 &center) :
		offset_(center),
		q_(rotation_quaternion),
		qi_(inverted(rotation_quaternion))
	{}

	template <typename TView>
	BOLT_DECL_HYBRID
	typename TView::Element operator()(
		const TView &view,
		Vector<typename TView::TIndex, 3> index) const
	{
		Float3 center_offset(0.5f, 0.5f, 0.5f);
		Float3 position(index);
		position -= offset_ - center_offset;
		position = (qi_ * position * q_).getVector() + offset_ - center_offset;
		return TInterpolator::compute(view, position);
	}
	//TODO(johny) - replace by matrix - more efficient than the quaternions for repeated computations
	Float3 offset_;
	Quaternion<float> q_, qi_;
};


/// Created rotated view - wrapper for TransformedImageView construction, so the template arguments can be autodeduced.
template<typename TView, typename TBorderHandlingTraits = BorderHandlingTraits<BorderHandling::kZero>>
TransformedImageView<TView, RotatedAccessOperator<LinearInterpolator<TBorderHandlingTraits>>>
rotatedView(
	TView view,
	const Quaternion<float> &rotation_quaternion,
	const Float3 &center,
	Vector<typename TView::TIndex, 3> size,
	TBorderHandlingTraits traits = BorderHandlingTraits<BorderHandling::kZero>())
{
	return TransformedImageView<TView, RotatedAccessOperator<LinearInterpolator<TBorderHandlingTraits>>>(
			view,
			size,
			RotatedAccessOperator<LinearInterpolator<TBorderHandlingTraits>>(
				rotation_quaternion,
				center));
}

//********************************************************************************************************************
//********************************************************************************************************************

namespace detail {
template<typename TInputView, typename TOutputView, typename TInverseTransformation>
struct TransformationFunctor
{
	BOLT_HD_WARNING_DISABLE
	template<typename TValue, typename TIndex>
	BOLT_DECL_HYBRID
	void operator()(TValue &value, TIndex index) const
	{
		value = input.access(transformation(coordinatesFromIndex(output, index)));
	}

	TInputView input;
	TOutputView output;
	TInverseTransformation transformation;
};

template<typename TInputView, typename TOutputView, typename TInverseTransformation>
TransformationFunctor<TInputView, TOutputView, TInverseTransformation>
makeTransformationFunctor(TInputView input, TOutputView output, TInverseTransformation inverse_transformation) {
	return TransformationFunctor<TInputView, TOutputView, TInverseTransformation>{ input, output, inverse_transformation };
}

}  // namespace detail

template<typename TInputView, typename TOutputView, typename TInverseTransformation>
void geometryTransformation(TInputView input, TOutputView output, TInverseTransformation inverse_transformation, cudaStream_t cuda_stream = 0) {
	static_assert(IsInterpolatedView<TInputView>::value, "Only interpolated views can be used as input for geometrical transformations.");
	forEachPosition(output, detail::makeTransformationFunctor(input, output, inverse_transformation), cuda_stream);
}

template<typename TScalingVector, typename TCoordinateType>
struct ScalingTransformation
{
	BOLT_DECL_HYBRID TCoordinateType
	operator()(const TCoordinateType &point) const
	{
		return offset + product(point - offset, scale);
	}
	TCoordinateType offset;
	TScalingVector scale;
};

template<typename TScalingVector, typename TCoordinateType>
ScalingTransformation<TScalingVector, TCoordinateType>
getInverseScaling(TCoordinateType anchor, TScalingVector scale_vector) {
	return ScalingTransformation<TScalingVector, TCoordinateType>{ anchor, div(TScalingVector::fill(1.0), scale_vector) };
}

template<typename TInputView, typename TOutputView, typename TOffsetCoordinates, typename TScaleVector>
void scale(TInputView input, TOutputView output, TOffsetCoordinates anchor, TScaleVector scale_vector, cudaStream_t cuda_stream = 0) {
	geometryTransformation(input, output, getInverseScaling(anchor, scale_vector), cuda_stream);
}

template<typename TInputView, typename TOutputView, typename TScaleVector>
void scale(TInputView input, TOutputView output, TScaleVector scale_vector) {
	geometryTransformation(input, output, getInverseScaling(Vector<float, TInputView::kDimension>(), scale_vector));
}

}  // namespace bolt
