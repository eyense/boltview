#pragma once

#include <initializer_list>
#include <ostream>

namespace bolt {

template<int... tInts >
class IntSequence {
	constexpr int Size() const {
		return sizeof...(tInts);
	}
};


namespace detail {

template<int tValue, int tHead, int... tTail>
struct IsValueInIntSequenceImpl {
	static constexpr bool kValue = (tValue == tHead) || IsValueInIntSequenceImpl<tValue, tTail...>::kValue;
};

template<int tValue, int tHead>
struct IsValueInIntSequenceImpl<tValue, tHead> {
	static constexpr bool kValue = tValue == tHead;
};

template<int tCurrentIndex, int tValue, int tHead, int... tTail>
struct GetPositionInIntSequenceImpl {
	static constexpr int kValue = tValue == tHead ? tCurrentIndex : GetPositionInIntSequenceImpl<tCurrentIndex + 1, tValue, tTail...>::kValue;
};

template<int tCurrentIndex, int tValue, int tHead>
struct GetPositionInIntSequenceImpl<tCurrentIndex, tValue, tHead> {
	static constexpr int kValue = tValue == tHead ? tCurrentIndex : -1;
};

template<int tFront, int...tValues>
struct GenerateIntSequenceImpl {
	using Type = typename GenerateIntSequenceImpl<tFront-1, tFront, tValues...>::Type;
};

template<int...tValues>
struct GenerateIntSequenceImpl<0, tValues...> {
	using Type = IntSequence<0, tValues...>;
};

}  // namespace detail


template<int tValue, typename TIntSequence>
struct IsValueInIntSequence {
	template<bool tDummy, typename TDummy>
	struct Helper;

	template<bool tDummy, int... tSequence>
	struct Helper<tDummy, IntSequence<tSequence...>> {
		static constexpr bool kValue = detail::IsValueInIntSequenceImpl<tValue, tSequence...>::kValue;
	};


	static constexpr bool kValue = Helper<false, TIntSequence>::kValue;
};


template<int tValue, typename TIntSequence>
struct GetPositionInIntSequence {
	static_assert(IsValueInIntSequence<tValue, TIntSequence>::kValue, "Value is not in the IntSequence!");

	template<bool tDummy, typename TDummy>
	struct Helper;

	template<bool tDummy, int... tSequence>
	struct Helper<tDummy, IntSequence<tSequence...>> {
		static constexpr int kValue = detail::GetPositionInIntSequenceImpl<0, tValue, tSequence...>::kValue;
	};


	static constexpr int kValue = Helper<false, TIntSequence>::kValue;
};

template<int tSize>
struct GenerateIntSequence {
	static_assert(tSize > 0, "Size of the IntSequence must be positive");
	using Type = typename detail::GenerateIntSequenceImpl<tSize - 1>::Type;
};

template<int tSize>
using MakeIntSequence = typename GenerateIntSequence<tSize>::Type;

template<int...tValues>
std::ostream &operator<<(std::ostream &stream, const IntSequence<tValues...> &) {
	stream << "[";

	(void) std::initializer_list<int>{((stream << tValues << ", "), 0)...};

	return stream << "]";
}

}  // namespace bolt

