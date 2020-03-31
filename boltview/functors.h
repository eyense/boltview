// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <algorithm>
#include <utility>

#include <boltview/tuple.h>
#include <boltview/math/complex.h>


namespace bolt {

/// Return the input unchanged
/// Usable in device/host code.
struct IdentityFunctor {
	template<typename TT>
	BOLT_DECL_HYBRID
	TT operator()(TT value) const {
		return value;
	}
};

/// Returns square of the input (i.e. input * input)
/// Usable in device/host code.
struct SquareFunctor {
	template<typename TT>
	BOLT_DECL_HYBRID
	auto operator()(TT value) const -> decltype(value * value) {
		return value * value;
	}
};

/// Returns square root of input
/// Usable in device/host code
struct SquareRootFunctor {
	template<typename TT>
	BOLT_DECL_HYBRID
	auto operator()(TT value) const -> decltype(sqrt(value)) {
		return sqrt(value);
	}
};

/// Returns input cast using C-style cast into TOutType
/// Usable in device/host code
template<typename TOutType>
struct CastFunctor {
	template<typename TT>
	BOLT_DECL_HYBRID
	TOutType operator()(TT value) const {
		return TOutType(value);
	}
};

template<typename TType>
struct ClampFunctor {
	BOLT_DECL_HYBRID
	TType operator()(TType value) const {
		return clamp(value, low, high);
	}

	TType low;
	TType high;
};

struct RoundFunctor {
	template<typename TT>
	BOLT_DECL_HYBRID
	auto operator()(TT value) const {
		return round(value);
	}
};

/// Multiplies passed value by factor specified in constructor.
/// Usable in device/host code.
template<typename TFactor>
struct MultiplyByFactorFunctor {
	explicit MultiplyByFactorFunctor(TFactor factor) :
		factor_(factor)
	{}

	template<typename TT>
	BOLT_DECL_HYBRID
	auto operator()(TT value) const -> decltype(std::declval<TFactor>() * value) {
		return factor_ * value;
	}

	TFactor factor_;
};

/// Adds specified constant to the passed value
/// Usable in device/host code.
template<typename TType>
struct AddValueFunctor {
	explicit AddValueFunctor(TType value) :
		value_(value)
	{}

	template<typename TT>
	BOLT_DECL_HYBRID
	auto operator()(TT in_value) const -> decltype(std::declval<TType>() + in_value) {
		return value_ + in_value;
	}

	TType value_;
};

/// Returns input incremented by a fixed value that is passes on construction
/// Usable in device/host code
template<typename TType>
struct IncrementFunctor {
	explicit IncrementFunctor(TType value) :
		value_(value)
	{}

	template<typename TT>
	BOLT_DECL_HYBRID
	void operator()(TT &in_value) const {
		in_value += value_;
	}

	TType value_;
};

struct MaxFunctor {
	template<typename TType>
	BOLT_DECL_HYBRID
	TType operator()(TType first, TType second) const {
			return max(first, second);
	}
};


/// Returns maximum of passed value and specified limit value
/// Usable in device/host code.
template<typename TType>
struct MaxWithLimitFunctor {
	explicit MaxWithLimitFunctor(TType limit) :
		limit_(limit)
	{}

	BOLT_DECL_HYBRID
	TType operator()(TType in_value) const {
		return max(limit_, in_value);
	}

	TType limit_;
};

/// Returns minumum of passed value and specified limit value
/// Usable in device/host code.
template<typename TType>
struct MinWithLimitFunctor {
	explicit MinWithLimitFunctor(TType limit) :
		limit_(limit)
	{}

	BOLT_DECL_HYBRID
	TType operator()(TType in_value) const {
		return min(limit_, in_value);
	}

	TType limit_;
};

/// If passed value is smaller than the limit return replacement value
/// Usable in device/host code.
template<typename TType>
struct LowerLimitFunctor {
	LowerLimitFunctor(TType limit, TType replacement) :
		limit_(limit),
		replacement_(replacement)
	{}

	BOLT_DECL_HYBRID
	TType operator()(TType in_value) const {
		return in_value < limit_ ? replacement_ : in_value;
	}

	TType limit_;
	TType replacement_;
};

/// If passed value is bigger than the limit return replacement value
/// Usable in device/host code.
template<typename TType>
struct UpperLimitFunctor {
	UpperLimitFunctor(TType limit, TType replacement) :
		limit_(limit),
		replacement_(replacement)
	{}

	BOLT_DECL_HYBRID
	TType operator()(TType in_value) const {
		return in_value > limit_ ? replacement_ : in_value;
	}

	TType limit_;
	TType replacement_;
};

/// Returns absolute value of the input.
/// Usable in device/host code
struct AbsFunctor {
	template<typename TType>
	BOLT_DECL_HYBRID
	TType operator()(TType in_value) const {
		return abs(in_value);
	}
};

/// Replaces the input with its absolute value
/// Usable in device/host code
struct AbsFunctorInplace {
	template<typename TType>
	BOLT_DECL_HYBRID
	void operator()(TType &in_value) const {
		in_value = AbsFunctor()(in_value);
	}
};

/// Returns minimum and maximum (in this order) of the input
/// Usable in device/host code
struct MinMaxFunctor {
	template<typename TType>
	BOLT_DECL_HYBRID
	Tuple<TType, TType> operator()(Tuple<TType, TType> first, Tuple<TType, TType> second) const {
		#ifdef BOLT_NVCC_KERNEL_CODE
			return Tuple<TType, TType>(min(first.template get<0>(), second.template get<0>()), max(first.template get<1>(), second.template get<1>()));
		#else
			return Tuple<TType, TType>(std::min(first.template get<0>(), second.template get<0>()), std::max(first.template get<1>(), second.template get<1>()));
		#endif
	}
};

/// Returns minimum, maximum and mean (in this order) of the input
/// Usable in device/host code
struct MinMaxMeanFunctor {
	template<typename TType, typename TMeanType>
	BOLT_DECL_HYBRID
	Tuple<TType, TType, TMeanType> operator()(Tuple<TType, TType, TMeanType> first, Tuple<TType, TType, TMeanType> second) const {
		#ifdef BOLT_NVCC_KERNEL_CODE
			return Tuple<TType, TType, TMeanType>(min(first.template get<0>(), second.template get<0>()), max(first.template get<1>(), second.template get<1>()), first.template get<2>() + second.template get<2>());
		#else
			return Tuple<TType, TType, TMeanType>(std::min(first.template get<0>(), second.template get<0>()), std::max(first.template get<1>(), second.template get<1>()), first.template get<2>() + second.template get<2>());
		#endif
	}
};

/// Returns a tuple formed from input parameters
/// Usable in device/host code
struct ZipValues
{
	template<typename... TTypes>
	BOLT_DECL_HYBRID Tuple<TTypes...>
	operator()(TTypes... a_args) const
	{
		return Tuple<TTypes...>(a_args...);
	}
};

/// Returns the normalized element
struct NormalizeFunctor {
	template <typename TComplexType>
	BOLT_DECL_HYBRID
	TComplexType operator()(TComplexType a) const {
		return normalize(a);
	}
};

/// Returns Magnitude of the complex number.
struct MagnitudeFunctor {
	template <typename TComplexType>
	BOLT_DECL_HYBRID
	float operator()(TComplexType in_value) const {
		return magnitude(in_value);
	}
};

/// Returns phase of the complex number.
struct PhaseFunctor {
	template <typename TComplexType>
	BOLT_DECL_HYBRID
	float operator()(TComplexType in_value) const {
		return phase(in_value);
	}
};


/// Returns the conjugate complex number.
struct ConjugateFunctor {
	template <typename TComplexType>
	BOLT_DECL_HYBRID
	TComplexType operator()(TComplexType in_value) const {
		return conjugate(in_value);
	}
};

/// Returns the hadamard product of complex numbers.
struct HadamardFunctor {
	template <typename TComplexType>
	BOLT_DECL_HYBRID
	TComplexType operator()(TComplexType lhs, TComplexType rhs) const {
		return hadamard(lhs, rhs);
	}
};

/// Usable in device/host code.
template<int tDimension>
struct PhaseShiftFunctor {
	using ShiftType = Vector<float, tDimension>;

	PhaseShiftFunctor(ShiftType pixel_shift, Vector<int, tDimension> size) : shift_(pixel_shift){
		for(int dim = 0; dim < tDimension; dim++){
			size_[dim] = static_cast<float>(size[dim]);
		}
	}

	PhaseShiftFunctor(ShiftType pixel_shift, int size) : shift_(pixel_shift){
		static_assert(tDimension == 1, "Only for 1D functors");
		size_[0] = static_cast<float>(size);
	}

	template<typename TComplexType, typename TIndex>
	BOLT_DECL_HYBRID
	auto operator()(TComplexType value, TIndex coordinates) const -> TComplexType {
		static_assert(TIndex::kDimension == tDimension, "Phase shift dimenstions and view dimensions must match");
		TComplexType shifter;
		int dim = 0;
		float arg = 2*kPi*coordinates[dim]*shift_[dim]/size_[dim];
		shifter.x = cos(arg);
		shifter.y = -sin(arg);
		return shifter * value;
	}

	private:
		ShiftType shift_;
		ShiftType size_;
};

}  // namespace bolt
