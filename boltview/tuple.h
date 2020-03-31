// Copyright 2017 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se
// Author: Ondrej Pacovsky op@eyen.se

// TODO(johny): This file needs a bit of a cleanup and linting.

#pragma once

namespace bolt {

namespace detail {


// helpers
template <typename TT>
struct Id { using type = TT; };

template <typename TT>
using TypeOf = typename TT::type;

template <size_t... tN>
struct Sizes : Id<Sizes<tN...>> { };

// choose N-th element in list <T...>
template <size_t tN, typename... TT>
struct Choose;

template <size_t tN, typename TH, typename... TT>
struct Choose<tN, TH, TT...> : Choose<tN-1, TT...> { };

template <typename TH, typename... TT>
struct Choose<0, TH, TT...> : Id<TH> { };

template <size_t tN, typename... TT>
using choose = TypeOf<Choose<tN, TT...>>;

// given L>=0, generate sequence <0, ..., L-1>
template <size_t tL, size_t tI = 0, typename TS = Sizes<>>
struct Range;

template <size_t tL, size_t tI, size_t... tN>
struct Range<tL, tI, Sizes<tN...>> : Range<tL, tI+1, Sizes<tN..., tI>> { };

template <size_t tL, size_t... tN>
struct Range<tL, tL, Sizes<tN...>> : Sizes<tN...> { };

template <size_t tL>
using RangeOf = TypeOf<Range<tL>>;

// single tuple element
BOLT_HD_WARNING_DISABLE
template <size_t tN, typename TT>
class TupleElem
{
	TT element_;
public:
	BOLT_HD_WARNING_DISABLE
	TupleElem() = default;

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	explicit TupleElem(TT element)
		: element_(element)
	{}

	BOLT_DECL_HYBRID
	TT&       get()       { return element_; }

	BOLT_DECL_HYBRID
	const TT& get() const { return element_; }
};

// tuple implementation
template <typename TN, typename... TT>
class TupleImpl;

template <size_t... tN, typename... TT>
class TupleImpl<Sizes<tN...>, TT...> : TupleElem<tN, TT>...
{
	template <size_t tM> using Pick = choose<tM, TT...>;
	template <size_t tM> using Elem = TupleElem<tM, choose<tM, TT...>>;

public:
	BOLT_HD_WARNING_DISABLE
	TupleImpl() = default;

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	explicit TupleImpl(TT... a_items)
		: TupleElem<tN, TT>(a_items)...
	{}

	template <size_t tM>
	BOLT_DECL_HYBRID
	Pick<tM>& get() { return Elem<tM>::get(); }

	template <size_t tM>
	BOLT_DECL_HYBRID
	const Pick<tM>& get() const { return Elem<tM>::get(); }
};

/// Helper struct for comparison operations on Tuple, borrowed from
/// http://stackoverflow.com/a/32236145
template<class TTupleT, std::size_t tJ, std::size_t tN>
struct TupleOps {
	BOLT_DECL_HYBRID
	static bool less(const TTupleT& x, const TTupleT& y) {
		return x.template get<tJ>() < y.template get<tJ>() ||
			( !(y.template get<tJ>() < x.template get<tJ>()) &&
				TupleOps<TTupleT, tJ + 1, tN>::less(x, y) );
	}
};

template <typename TTuple, size_t tN>
struct TupleOps<TTuple, tN, tN> {
	BOLT_DECL_HYBRID
	static bool less(const TTuple& /*unused*/, const TTuple& /*unused*/) {return false;}
};
}  // namespace detail


/// \addtogroup Math
/// @{

/// Tuple usable both in device and host code.
/// Use instead of std::tuple in CUDA programming.
template <typename... TT>
struct Tuple : detail::TupleImpl<detail::RangeOf<sizeof...(TT)>, TT...>
{
	using Predecessor = detail::TupleImpl<detail::RangeOf<sizeof...(TT)>, TT...>;

	Tuple() = default;

	BOLT_DECL_HYBRID
	explicit Tuple(TT... items)
		: Predecessor(items...)
	{}

	BOLT_DECL_HYBRID
	static constexpr std::size_t size() { return sizeof...(TT); }
};

template<class... TT>
BOLT_DECL_HYBRID
bool operator<(const Tuple<TT...>& x, const Tuple<TT...>& y) {
	return detail::TupleOps<decltype(x), 0, sizeof...(TT)>::less(x, y);
}


/// @}

/*template<int tIdx, typename... TType>
struct Get_policy<tIdx, const Tuple<TType...>>
{
	using return_type = const choose<tIdx, TType...> &;
	using value_t = const Tuple<TType...> &;

	static BOLT_DECL_HYBRID auto
	get(value_t aArg) -> decltype(aArg.template get<tIdx>())
	{
		return aArg.template get<tIdx>();
	}
};

template<int tIdx, typename... TType>
struct Get_policy<tIdx, Tuple<TType...>>
{
	using return_type = choose<tIdx, TType...> &;
	using value_t = Tuple<TType...> &;

	static BOLT_DECL_HYBRID auto
	get(value_t aArg) -> decltype(aArg.template get<tIdx>())
	{
		return aArg.template get<tIdx>();
	}
};*/

}  // namespace bolt
