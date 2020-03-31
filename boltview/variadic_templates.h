#pragma once

#include <initializer_list>
#include <boltview/int_sequence.h>

namespace bolt {

template<typename ...TArgs>
void IgnoreReturnValues(TArgs ...) {}

namespace detail {

}  // namespace detail

/// Find type T in list of types.
/// Check core library examples.
template <typename... >
struct Ordinal : std::integral_constant<int, -1> { };

// found it
template <typename T, typename... R>
struct Ordinal<T, T, R...> : std::integral_constant<int, 0>
{ };

// still looking
template <typename T, typename F, typename... R>
struct Ordinal<T, F, R...> : std::integral_constant<int, 1 + Ordinal<T,R...>::value> {
	static_assert(sizeof...(R) > 0, "Requested type is not present in the type list");
};


/// get type from type list by index
/// Check core library examples.
template <int tIndex, typename... TTypes>
struct Index {
	static_assert(tIndex < 0 || sizeof...(TTypes) > 0, "Index bigger then the type list size");
	static_assert(tIndex >= 0 || sizeof...(TTypes) > 0, "Index is negative");
	using type = void;
};

template <typename THead, typename... TTail>
struct Index<0, THead, TTail...> {
	using type = THead;
};

template <int tIndex, typename THead, typename... TTail>
struct Index<tIndex, THead, TTail...> : Index<tIndex - 1, TTail...> {
};


namespace detail {

template <typename TCallable, typename TTuple, int... tI>
auto applyImpl(TCallable&& f, TTuple&& t, bolt::IntSequence<tI...>) -> typename std::result_of<TCallable>::type
{
	return f(std::get<tI>(std::forward<TTuple>(t))...);
}

}  // namespace detail

/// Call callable with parameters passed via tuple
template <typename TCallable, typename TTuple>
auto apply(TCallable&& f, TTuple&& t) -> typename std::result_of<TCallable>::type
{
	static constexpr int kTupleSize = std::tuple_size<typename std::remove_reference<TTuple>::type>::value;
	return detail::applyImpl(std::forward<TCallable>(f), std::forward<TTuple>(t), bolt::MakeIntSequence<kTupleSize>{});
}


template<class F, class A0>
auto fold(F&&, A0&& a0) {
    return std::forward<A0>(a0);
}

template<class F, class A0, class...As>
auto fold(F&& f, A0&&a0, As&&...as) {
    return f(std::forward<A0>(a0), fold(f, std::forward<As>(as)...));
}



}  // namespace bolt

