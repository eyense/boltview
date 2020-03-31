// Copyright 2017 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <boltview/cuda_defines.h>
#include <boltview/device_image_view_base.h>
#include <boltview/host_image_view_base.h>
#include <boltview/exceptions.h>
#include <boltview/functors.h>
#include <boltview/tuple.h>
#include <boltview/interpolation.h>

namespace bolt {

/// Procedural image view, which returns same value for all indices.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class ConstantImageView : public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsDeviceView = true;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = false;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using IndexType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = TElement;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	ConstantImageView(TElement element, SizeType size) :
		Predecessor(size),
		element_(element)
	{}

	BOLT_DECL_HYBRID
	TElement operator[](IndexType /*index*/) const {
		return element_;
	}

protected:
	Element element_;
};

/// Utility function to create ConstantImageView without the need to specify template parameters.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
ConstantImageView<TElement, tDimension, TPolicy>
makeConstantImageView(TElement value, Vector<typename TPolicy::IndexType, tDimension> size, TPolicy  /*unused*/= TPolicy()) {
	return ConstantImageView<TElement, tDimension, TPolicy>(value, size);
}

template<typename TElement, typename TPolicy = DefaultViewPolicy>
ConstantImageView<TElement, 1> makeConstantImageView(TElement value, size_t size, TPolicy  /*unused*/= TPolicy()) {
	return ConstantImageView<TElement, 1, TPolicy>(value, size);
}


/// Procedural image view, which returns same value for all indices.
template<int tDimension, typename TId = int, typename TPolicy = DefaultViewPolicy>
class UniqueIdImageView : public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsDeviceView = true;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = false;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = TId;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	explicit UniqueIdImageView(SizeType size, TId first = 0) :
		Predecessor(size),
		first_(first)
	{}

	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return first_ + static_cast<Element>(getLinearAccessIndex(this->size(), index));
	}

protected:
	Element first_;
};

/// Utility function to create ConstantImageView without the need to specify template parameters.
template<int tDimension, typename TId, typename TPolicy = DefaultViewPolicy>
UniqueIdImageView<tDimension, TId>
makeUniqueIdImageView(Vector<typename TPolicy::IndexType, tDimension> size, TId first, TPolicy  /*unused*/= TPolicy()) {
	return UniqueIdImageView<tDimension, TId, TPolicy>(size, first);
}
/// Procedural image view, which generates checker board like image.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
class CheckerBoardImageView : public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsDeviceView = true;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = false;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = TElement;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	CheckerBoardImageView(TElement white, TElement black, SizeType tile_size, SizeType size) :
		Predecessor(size),
		tile_size_(tile_size),
		white_(white),
		black_(black)
	{}

	BOLT_DECL_HYBRID
	TElement operator[](IndexType index) const {
		return sum(div(index, tile_size_)) % 2 ? white_ : black_;
	}

protected:
	SizeType tile_size_;
	Element white_;
	Element black_;
};


/// Utility function to create CheckerBoardImageView without the need to specify template parameters.
template<typename TElement, int tDimension, typename TPolicy = DefaultViewPolicy>
CheckerBoardImageView<TElement, tDimension, TPolicy>
checkerboard(
			TElement white,
			TElement black,
			Vector<typename TPolicy::IndexType, tDimension> tile_size,
			Vector<typename TPolicy::IndexType, tDimension> size,
			TPolicy  /*unused*/= TPolicy())
{
	return CheckerBoardImageView<TElement, tDimension, TPolicy>(white, black, tile_size, size);
}

// Procedural image view, which generates samples (according to size) of one period of sinus wave independently in each dimension and sums the result
template<int tDimension>
class SinusImageView : public HybridImageViewBase<tDimension> {
public:
	static const bool kIsDeviceView = true;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = false;
	static const int kDimension = tDimension;
	using SizeType = Vector<int, tDimension>;
	using IndexType = Vector<int, tDimension>;
	using DataType = Vector<float, tDimension>;
	using Predecessor = HybridImageViewBase<tDimension>;
	using Element = float;
	using AccessType = Element;

	BOLT_DECL_HYBRID
	SinusImageView(SizeType size, DataType amplitude, DataType frequency, DataType phase) :
		Predecessor(size),
		amplitude_(amplitude),
		frequency_(frequency),
		phase_(phase)
	{}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		float result = 0;
		for(int dim = 0; dim < tDimension; dim++){
			float x = lerp(static_cast<float>(index[dim]) / static_cast<float>(get(this->size_, dim)), .0, 2*kPi);
			result += amplitude_[dim] * sin(frequency_[dim]*x + phase_[dim]);
		}
		return result;
	}

protected:
	DataType amplitude_;
	DataType frequency_;
	DataType phase_;
};

/// Utility function to create SinusImageView without the need to specify template parameters.
template<int tDimension>
SinusImageView<tDimension>
makeSinusImageView(Vector<int, tDimension> size, Vector<float, tDimension> amplitude, Vector<float, tDimension> frequency, Vector<float, tDimension> phase)
{
	return SinusImageView<tDimension>(size, amplitude, frequency, phase);
}

/// Base class for image views implementing lazy evaluation of operators working on two image views.
template<typename TView1, typename TView2>
class BinaryOperatorImageViewBase : public HybridImageViewBase<TView1::kDimension, typename TView1::Policy> {
public:
	static_assert(TView1::kDimension == TView2::kDimension, "Both views must have same dimension!");
	static const bool kIsDeviceView = TView1::kIsDeviceView && TView2::kIsDeviceView;
	static const bool kIsHostView = TView1::kIsHostView && TView2::kIsHostView;
	static_assert(kIsDeviceView || kIsHostView, "Both views must be usable either in device code or in host code");
	// TODO(martin): Test compatibility of policies.
	// The following assert does not work because of ArrayViews
	// static_assert(std::is_same<typename TView1::Policy, typename TView2::Policy>::value, "Both views must have the same policy");
	static const int kDimension = TView1::kDimension;
	using Policy = typename TView1::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;
	using Predecessor = HybridImageViewBase<kDimension, Policy>;

	using Element1 = typename TView1::Element;
	using Element2 = typename TView2::Element;

	BinaryOperatorImageViewBase(TView1 view1, TView2 view2) :
		Predecessor(view1.size()),
		view1_(view1),
		view2_(view2)
	{
		/* TODO(johny) if (view1.size() != view2.size()) {
			BOLT_THROW(EIncompatibleViewSizes());
		}*/
	}

protected:
	TView1 view1_;
	TView2 view2_;
};


/// Image view, which returns result of a functor
template<typename TView1, typename TView2, typename TFunctor>
class BinaryFunctorImageView : public BinaryOperatorImageViewBase<TView1, TView2> {
public:
	static const int kDimension = BinaryOperatorImageViewBase<TView1, TView2>::kDimension;
	using SizeType = typename BinaryOperatorImageViewBase<TView1, TView2>::SizeType;
	using IndexType = typename BinaryOperatorImageViewBase<TView1, TView2>::IndexType;
	using Predecessor = BinaryOperatorImageViewBase<TView1, TView2>;

	using Element1 = typename TView1::Element;
	using Element2 = typename TView2::Element;

	using Element = decltype(std::declval<TFunctor>()(std::declval<Element1>(), std::declval<Element2>()));

	using AccessType = Element;

	BinaryFunctorImageView(TView1 view1, TView2 view2, TFunctor functor) :
		Predecessor(view1, view2),
		functor_(functor)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return functor_(this->view1_[index], this->view2_[index]);
	}

protected:
	TFunctor functor_;
};

template<typename TView1, typename TView2>
BinaryFunctorImageView<TView1, TView2, HadamardFunctor>
hadamard(TView1 view1, TView2 view2) {
	return BinaryFunctorImageView<TView1, TView2, HadamardFunctor>(view1, view2, HadamardFunctor());
}

/// Image view, which returns linear combination of elements from two other image views.
/// R = f1 * I1 + f2 * I2
template<typename TFactor1, typename TView1, typename TFactor2, typename TView2>
class LinearCombinationImageView : public BinaryOperatorImageViewBase<TView1, TView2> {
public:
	static const int kDimension = TView1::kDimension;
	using Policy = typename TView1::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;
	using Predecessor = BinaryOperatorImageViewBase<TView1, TView2>;

	using Element1 = typename TView1::Element;
	using Element2 = typename TView2::Element;

	using Result1 = decltype(std::declval<TFactor1>() * std::declval<Element1>());
	using Result2 = decltype(std::declval<TFactor2>() * std::declval<Element2>());
	using Element = decltype(std::declval<Result1>() + std::declval<Result2>());
	using AccessType = Element;

	LinearCombinationImageView(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) :
		Predecessor(view1, view2),
		factor1_(factor1),
		factor2_(factor2)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return factor1_ * this->view1_[index] + factor2_ * this->view2_[index];
	}

protected:
	TFactor1 factor1_;
	TFactor2 factor2_;
};


/// Utility function to create LinearCombinationImageView without the need to specify template parameters.
template<typename TFactor1, typename TView1, typename TFactor2, typename TView2>
LinearCombinationImageView<TFactor1, TView1, TFactor2, TView2>
linearCombination(TFactor1 factor1, TView1 view1, TFactor2 factor2, TView2 view2) {
	return LinearCombinationImageView<TFactor1, TView1, TFactor2, TView2>(factor1, view1, factor2, view2);
}


/// Utility function wrapping two image addition.
template<typename TView1, typename TView2>
LinearCombinationImageView<int, TView1, int, TView2>
add(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationImageView<int, TView1, int, TView2>(1, view1, 1, view2);
}


/// Utility function wrapping two image subtraction.
template<typename TView1, typename TView2>
LinearCombinationImageView<int, TView1, int, TView2>
subtract(TView1 view1, TView2 view2) {
	// TODO(johny) - possible more efficient implementation
	return LinearCombinationImageView<int, TView1, int, TView2>(1, view1, -1, view2);
}


/// Image view, which returns per element multiplication.
/// R = I1 .* I2
template<typename TView1, typename TView2>
class MultiplicationImageView : public BinaryOperatorImageViewBase<TView1, TView2> {
public:
	static const int kDimension = TView1::kDimension;
	using Policy = typename TView1::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;
	using Predecessor = BinaryOperatorImageViewBase<TView1, TView2>;

	using Element1 = typename TView1::Element;
	using Element2 = typename TView2::Element;


	using Element = decltype(std::declval<Element1>() * std::declval<Element2>());
	using AccessType = Element;

	MultiplicationImageView(TView1 view1, TView2 view2) :
		Predecessor(view1, view2)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return this->view1_[index] * this->view2_[index];
	}
};


/// Utility function to create MultiplicationImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
MultiplicationImageView<TView1, TView2>
multiply(TView1 view1, TView2 view2) {
	return MultiplicationImageView<TView1, TView2>(view1, view2);
}


/// Image view, which returns per element division.
/// R = I1 ./ I2
template<typename TView1, typename TView2>
class DivisionImageView : public BinaryOperatorImageViewBase<TView1, TView2> {
public:
	static const bool kIsDeviceView = true;
	static const int kDimension = TView1::kDimension;
	using Policy = typename TView1::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;
	using Predecessor = BinaryOperatorImageViewBase<TView1, TView2>;

	using Element1 = typename TView1::Element;
	using Element2 = typename TView2::Element;

	using Element = decltype(std::declval<Element1>() / std::declval<Element2>());
	using AccessType = Element;

	DivisionImageView(TView1 view1, TView2 view2) :
		Predecessor(view1, view2)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return this->view1_[index] / this->view2_[index];
	}
};


/// Utility function to create DivisionImageView without the need to specify template parameters.
template<typename TView1, typename TView2>
DivisionImageView<TView1, TView2>
divide(TView1 view1, TView2 view2) {
	return DivisionImageView<TView1, TView2>(view1, view2);
}


/// View which allows mirror access to another view
/// TODO(johny) - specialization for memory based views - only stride and pointer reordering
template<typename TView>
class MirrorImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, typename TView::Policy>;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;

	MirrorImageView(TView view, Vector<bool, kDimension> flips) :
		Predecessor(view.size()),
		view_(view),
		flips_(flips)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		return view_[flipCoordinates(index, this->size(), flips_)];
	}

protected:
	BOLT_DECL_HYBRID
	static IndexType flipCoordinates(IndexType index, SizeType size, Vector<bool, kDimension> flips) {
		for (int i = 0; i < kDimension; ++i) {
			if (flips[i]) {
				index[i] = size[i] - index[i] - 1;
			}
		}
		return index;
	}

	TView view_;
	Vector<bool, kDimension> flips_;
};


/// Create mirror views with fliped axes specification
template<typename TView>
MirrorImageView<TView>
mirror(TView view, Vector<bool, TView::kDimension> flips) {
	return MirrorImageView<TView>(view, flips);
}


/// View which pads another image view
/// TODO(johny) - do also tapper padding
template<typename TView/*, bool tIsPeriodic*/>
class PaddedImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, typename TView::Policy>;
	using Element = typename TView::Element;
	using AccessType = typename TView::Element;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;

	PaddedImageView(TView view, const SizeType &size, const SizeType &offset, Element fill_value) :
		Predecessor(size),
		view_(view),
		offset_(offset),
		fill_value_(fill_value)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		index = modPeriodic(index - offset_, this->size());
		if (view_.isIndexInside(index)) {
			return view_[index];
		}
		return fill_value_;
	}

protected:
	TView view_;
	SizeType offset_;
	Element fill_value_;
};


/// Create padded view
template<typename TView>
PaddedImageView<TView> paddedView(
	TView view,
	Vector<typename TView::TIndex, TView::kDimension> size,
	Vector<typename TView::TIndex, TView::kDimension> offset,
	typename TView::Element fill_value)
{
	return PaddedImageView<TView>(view, size, offset, fill_value);
}


/// View which applies unary operation for every element access and returns the output
template<typename TView, typename TOperator>
class UnaryOperatorImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Input = typename TView::Element;
	using Element = decltype(std::declval<TOperator>()(std::declval<Input>()));
	using AccessType = Element;

	UnaryOperatorImageView(TView view, TOperator unary_operator) :
		Predecessor(view.size()),
		view_(view),
		unary_operator_(unary_operator)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		return unary_operator_(view_[index]);
	}

protected:
	TView view_;
	TOperator unary_operator_;
};



/// View which applies unary operation for every element access and returns the output
template<typename TView, typename TOperator>
class UnaryOperatorWithMetadataImageView : public HybridImageViewBase<TView::kDimension> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension>;
	using Input = typename TView::Element;
	using Element = decltype(std::declval<TOperator>()(std::declval<Input>(), std::declval<IndexType>()));
	using AccessType = Element;

	UnaryOperatorWithMetadataImageView(TView view, TOperator unary_operator) :
		Predecessor(view.size()),
		view_(view),
		unary_operator_(unary_operator)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		return unary_operator_(view_[index], index);
	}

protected:
	TView view_;
	TOperator unary_operator_;
};



/// Creates view which returns casted values from the original view
template<typename TOutType, typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
UnaryOperatorImageView<TView, CastFunctor<TOutType>>
cast(TView view) {
	return UnaryOperatorImageView<TView, CastFunctor<TOutType>>(view, CastFunctor<TOutType>());
}

template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
auto round(TView view) {
	return UnaryOperatorImageView<TView, RoundFunctor>(view, RoundFunctor{});
}

template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
auto clamp(TView view, typename TView::Element low, typename TView::Element high) {
	return UnaryOperatorImageView<TView, ClampFunctor<typename TView::Element>>(
			view,
			ClampFunctor<typename TView::Element>{ low, high });
}


/// Returns per-element normalized elements form original view
template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
auto normalize(TView view) {
	return UnaryOperatorImageView<TView, NormalizeFunctor>(view, NormalizeFunctor());
}


/// Creates view which returns squared values from the original view
template<typename TView, class = typename std::enable_if<IsImageView<TView>::value>::type>
UnaryOperatorImageView<TView, SquareFunctor>
square(TView view) {
	return UnaryOperatorImageView<TView, SquareFunctor>(view, SquareFunctor());
}


/// Creates view which returns square root of the values from the original view
template<typename TView>
UnaryOperatorImageView<TView, SquareRootFunctor>
squareRoot(TView view) {
	return UnaryOperatorImageView<TView, SquareRootFunctor>(view, SquareRootFunctor());
}


/// Utility function to create multiplied view without the need to specify template parameters.
template<typename TFactor, typename TView>
UnaryOperatorImageView<TView, MultiplyByFactorFunctor<TFactor>>
multiplyByFactor(TFactor factor, TView view) {
	return UnaryOperatorImageView<TView, MultiplyByFactorFunctor<TFactor>>(view, MultiplyByFactorFunctor<TFactor>(factor));
}

/// Creates view returning values from the original view with value added.
template<typename TType, typename TView>
UnaryOperatorImageView<TView, AddValueFunctor<TType>>
addValue(TType value, TView view) {
	return UnaryOperatorImageView<TView, AddValueFunctor<TType>>(view, AddValueFunctor<TType>(value));
}


/// Returns view returning values from the original view with values lower then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorImageView<TView, MaxWithLimitFunctor<TType>>
lowerLimit(TType limit, TView view) {
	return UnaryOperatorImageView<TView, MaxWithLimitFunctor<TType>>(view, MaxWithLimitFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values lower then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorImageView<TView, LowerLimitFunctor<TType>>
lowerLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorImageView<TView, LowerLimitFunctor<TType>>(view, LowerLimitFunctor<TType>(limit, replacement));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the limit
template<typename TType, typename TView>
UnaryOperatorImageView<TView, MinWithLimitFunctor<TType>>
upperLimit(TType limit, TView view) {
	return UnaryOperatorImageView<TView, MinWithLimitFunctor<TType>>(view, MinWithLimitFunctor<TType>(limit));
}


/// Returns view returning values from the original view with values bigger then limit replaced by the specified replacement
template<typename TType, typename TView>
UnaryOperatorImageView<TView, UpperLimitFunctor<TType>>
upperLimit(TType limit, TType replacement, TView view) {
	return UnaryOperatorImageView<TView, UpperLimitFunctor<TType>>(view, UpperLimitFunctor<TType>(limit, replacement));
}

/// Returns view returning absolute values from the original view
template<typename TView>
UnaryOperatorImageView<TView, AbsFunctor>
absolute(TView view) {
	return UnaryOperatorImageView<TView, AbsFunctor>(view, AbsFunctor());
}

/// Returns magnitude of complex number from the original view
template<typename TView>
UnaryOperatorImageView<TView, MagnitudeFunctor>
magnitude(TView view) {
	return UnaryOperatorImageView<TView, MagnitudeFunctor>(view, MagnitudeFunctor());
}

/// Returns magnitude of complex number from the original view
template<typename TView>
UnaryOperatorImageView<TView, PhaseFunctor>
phase(TView view) {
	return UnaryOperatorImageView<TView, PhaseFunctor>(view, PhaseFunctor());
}

/// Returns magnitude of complex number from the original view
template<typename TView>
UnaryOperatorImageView<TView, ConjugateFunctor>
conjugate(TView view) {
	return UnaryOperatorImageView<TView, ConjugateFunctor>(view, ConjugateFunctor());
}


/// View returning single coordinate mapping from grid
/// Inspired by Matlab function 'meshgrid'
template<int tDimension, typename TPolicy = DefaultViewPolicy>
class MeshGridView: public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, tDimension>;
	using IndexType = Vector<TIndex, tDimension>;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = int;
	using AccessType = Element;

	MeshGridView() :
		Predecessor(SizeType())
	{}

	MeshGridView(IndexType from, IndexType to, int dimension) :
		Predecessor(abs(to - from)),
		dimension_(dimension),
		start_(from[dimension]),
		increment_(signum(to[dimension] - from[dimension_]))
	{}

	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		return start_ + index[dimension_] * increment_;
	}

protected:
	int dimension_ = 0;
	int start_ = 0;
	int increment_ = 0;
};

/// Rectangular grid in N-D space.
/// Inspired by Matlab function of the same name:
/// [X,Y] = MeshGrid(Int2(1, 10), Int2(3, 14))
/// X =
/// 	  1	 2	 3
/// 	  1	 2	 3
/// 	  1	 2	 3
/// 	  1	 2	 3
/// 	  1	 2	 3
/// Y =
///     10    10    10
///     11    11    11
///     12    12    12
///     13    13    13
///     14    14    14
template<int tDimension, typename TPolicy = DefaultViewPolicy>
std::array<MeshGridView<tDimension, TPolicy>, tDimension>
meshGrid(
	Vector<typename TPolicy::IndexType, tDimension> from,
	Vector<typename TPolicy::IndexType, tDimension> to,
	TPolicy  /*unused*/= TPolicy())
{
	std::array<MeshGridView<tDimension, TPolicy>, tDimension> result;
	for (int i = 0; i < tDimension; ++i) {
		result[i] = MeshGridView<tDimension, TPolicy>(from, to, i);
	}
	return result;
}

// TODO(johny) - C++17 - compile-time folds with initializer list expansion to remove the recursion
template<bool tFirstValue, bool ...tValues>
struct CompileTimeFoldAnd {
	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = tFirstValue && CompileTimeFoldAnd<tValues...>::value;
};

template<bool tValue>
struct CompileTimeFoldAnd<tValue> {
	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = tValue;
};

template<typename TFirstView, typename... TViews>
struct MultiViewTraits
{
	static constexpr int kDimension = TFirstView::kDimension;

	static constexpr bool kIsDeviceView = CompileTimeFoldAnd<TFirstView::kIsDeviceView, TViews::kIsDeviceView...>::value;
	static constexpr bool kIsHostView = CompileTimeFoldAnd<TFirstView::kIsHostView, TViews::kIsHostView...>::value;
};


template<typename TOperator, typename TView, typename... TViews>
class NAryOperatorImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	using MultiTraits = MultiViewTraits<TView, TViews...>;

	static const bool kIsDeviceView = MultiTraits::kIsDeviceView;
	static const bool kIsHostView = MultiTraits::kIsHostView;
	static const int kDimension = MultiTraits::kDimension;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = Vector<TIndex, kDimension>;
	using IndexType = Vector<TIndex, kDimension>;
	using Predecessor = HybridImageViewBase<kDimension, Policy>;
	using Element = decltype(std::declval<TOperator>()(std::declval<typename TView::Element>(), std::declval<typename TViews::Element>()...));
	using AccessType = Element;

	NAryOperatorImageView(TOperator op, TView view, TViews... views) :
		Predecessor(view.size()),
		mOperator(op),
		mViews(view, views...)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const
	{
		return call(index, typename detail::Range<1 + sizeof...(TViews)>::type());
	}

protected:

	BOLT_HD_WARNING_DISABLE
	template<size_t ...tN>
	BOLT_DECL_HYBRID
	Element call(IndexType index, detail::Sizes<tN...> /*indices*/) const
	{
		return mOperator(mViews.template get<tN>()[index]...);
	}

	TOperator mOperator;
	Tuple<TView, TViews...> mViews;
};


template<typename TFunctor, typename TView, typename... TViews>
NAryOperatorImageView<TFunctor, TView, TViews...>
nAryOperator(TFunctor functor, TView view, TViews... views) {
	return NAryOperatorImageView<TFunctor, TView, TViews...>(functor, view, views...);
}

template<typename TView, typename... TViews>
NAryOperatorImageView<ZipValues, TView, TViews...>
zipViews(TView view, TViews... views) {
	return NAryOperatorImageView<ZipValues, TView, TViews...>(ZipValues(), view, views...);
}

/// Procedural image view, which returns the index value for all indices.
template <int tDimension, typename TPolicy = DefaultViewPolicy>
class IndexView : public HybridImageViewBase<tDimension, TPolicy> {
public:
	static const bool kIsDeviceView = true;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = false;
	static const int kDimension = tDimension;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using IndexType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using Predecessor = HybridImageViewBase<tDimension, Policy>;
	using Element = IndexType;
	using AccessType = IndexType;

	BOLT_DECL_HYBRID
	explicit IndexView(SizeType size) : Predecessor(size) {}

	BOLT_DECL_HYBRID
	IndexType operator[](IndexType index) const { return index; }
};
/// @}


template <typename TAccessType, typename TIndexType>
struct ElementPositionPair {
public:
	BOLT_DECL_HYBRID
	ElementPositionPair(TAccessType ele, TIndexType pos) : element(ele), position(pos) {}

	ElementPositionPair() = default;

	template<typename TAccessType2, typename TIndexType2>
	BOLT_DECL_HYBRID
	explicit ElementPositionPair(const ElementPositionPair<TAccessType2, TIndexType2> &other) :
		element(other.element),
		position(other.position)
	{}

	template<typename TAccessType2, typename TIndexType2>
	BOLT_DECL_HYBRID
	ElementPositionPair<TAccessType, TIndexType> &
	operator=(const ElementPositionPair<TAccessType2, TIndexType2> &other) {
		element = other.element;
		position = other.position;
		return *this;
	}


	TAccessType element;
	TIndexType position;
};


template <typename TView>
class ElementPositionView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Element = ElementPositionPair<typename TView::Element, IndexType>;
	using AccessType = ElementPositionPair<typename TView::AccessType, IndexType>;

	explicit ElementPositionView(TView view) : Predecessor(view.size()), view_(view) {}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID AccessType operator[](IndexType index) const { return AccessType(view_[index], index); }

protected:
	TView view_;
};


template <typename TView>
ElementPositionView<TView> zipWithPosition(TView view) {
	return ElementPositionView<TView>(view);
}


}  // namespace bolt

#include <boltview/operators.tcc>
