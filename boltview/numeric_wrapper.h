#pragma once

#include <iostream>
#include <boost/serialization/nvp.hpp>


namespace bolt {

template<typename TType, typename TTag>
class NumericTypeWrapper {
public:
	struct Uninitialized {};
	constexpr NumericTypeWrapper() :
		value_(0)
	{}

	constexpr explicit NumericTypeWrapper(Uninitialized)
	{}

	constexpr explicit NumericTypeWrapper(TType value) :
		value_(value)
	{}

	constexpr NumericTypeWrapper(const NumericTypeWrapper &) = default;
	NumericTypeWrapper &operator=(const NumericTypeWrapper &) = default;

	constexpr TType get() const {
		return value_;
	}

	TType &get() {
		return value_;
	}

	constexpr bool operator==(const NumericTypeWrapper &value) const {
		return value_ == value.value_;
	}

	constexpr bool operator!=(const NumericTypeWrapper &value) const {
		return value_ != value.value_;
	}

protected:
	TType value_;
};

template<typename TType, typename TTag>
std::ostream &operator<<(std::ostream &stream, const NumericTypeWrapper<TType, TTag> &value) {
	return stream << value;
}

}  // namespace bolt


namespace boost {
namespace serialization {


template<class Archive, typename TType, typename TTag>
void serialize(Archive &ar, bolt::NumericTypeWrapper<TType, TTag> &wrapper, const unsigned int version) {
	// TODO(johny) - check if tag is correct?
	ar & boost::serialization::make_nvp("value", wrapper.get());
}

} // namespace serialization
} // namespace boost

