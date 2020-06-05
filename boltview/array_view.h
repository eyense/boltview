// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once
#include <type_traits>
#include <utility>

#include <thrust/device_vector.h>

#include <boltview/cuda_utils.h>
#include <boltview/exceptions.h>
#include <boltview/cuda_defines.h>
#include <boltview/view_policy.h>
#include <boltview/view_traits.h>

#include <boost/serialization/nvp.hpp>
#include <boost/serialization/array.hpp>

namespace bolt {

#if defined(__CUDACC__)
template<typename TElement>
class DeviceArrayView;

template<typename TElement>
class DeviceArrayConstView {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = 1;
	using SizeType = int64_t;
	using IndexType = int64_t;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;
	using Policy = LongIndexViewPolicy;
	using TIndex = typename Policy::IndexType;

	DeviceArrayConstView() = default;

	BOLT_DECL_HYBRID
	DeviceArrayConstView(const Element *pointer, SizeType size) :
		pointer_(pointer),
		size_(size)
	{}

	BOLT_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayConstView<Element> &view) :
		pointer_(view.pointer()),
		size_(view.size())
	{}

	BOLT_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayConstView<const Element> &view):
		pointer_(view.pointer_),
		size_(view.size_)
	{}

	BOLT_DECL_HYBRID
	DeviceArrayConstView(const DeviceArrayView<const Element> &view);

	BOLT_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayConstView<Element> &view) {
		pointer_ = view.pointer_;
		size_ = view.size_;
		return *this;
	}

	BOLT_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayConstView<const Element> &view) {
		pointer_ = view.pointer_;
		size_ = view.size_;
		return *this;
	}

	BOLT_DECL_HYBRID
	DeviceArrayConstView<TElement> &operator=(const DeviceArrayView<Element> &view);

	BOLT_DECL_HYBRID
	SizeType size()  const {
		return size_;
	}

	BOLT_DECL_HYBRID
	SizeType elementCount()  const {
		return size_;
	}

	BOLT_DECL_HYBRID
	bool empty() const {
		return size_ == 0;
	}

	BOLT_DECL_HYBRID
	const Element *pointer() const {
		return pointer_;
	}

	BOLT_DECL_DEVICE
	AccessType operator[](IndexType index) const {
		return pointer_[index];
	}

	/// Creates view for part of this view.
	DeviceArrayConstView<TElement> subview(IndexType from, SizeType size) const {
		BOLT_ASSERT(from >= 0);
		BOLT_ASSERT(size <= size_);
		BOLT_ASSERT(from + size <= size_);

		return DeviceArrayConstView(pointer_ + from, size);
	}

	int strides() const {
		return 1;
	}

protected:
	const Element *pointer_ = nullptr;
	SizeType size_ = 0;
};


template<typename TElement>
class DeviceArrayView: public DeviceArrayConstView<TElement> {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = 1;
	using SizeType = int64_t;
	using IndexType = int64_t;
	using Predecessor = DeviceArrayConstView<TElement>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element &;
	using Policy = LongIndexViewPolicy;
	using TIndex = typename Policy::IndexType;

	DeviceArrayView() = default;

	BOLT_DECL_HYBRID
	DeviceArrayView(TElement *pointer, SizeType size) :
		Predecessor(pointer, size)
	{}


	DeviceArrayView(const DeviceArrayView &) = default;

	DeviceArrayView &operator=(const DeviceArrayView &) = default;

	BOLT_DECL_DEVICE
	AccessType operator[](IndexType index) const {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_)[index];
	}

	BOLT_DECL_HYBRID
	Element *pointer() {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_);
	}

	BOLT_DECL_HYBRID
	const Element *pointer() const {
		return this->pointer_;
	}

	/// Creates view for part of this view.
	DeviceArrayView<TElement> subview(IndexType from, SizeType size) const {
		BOLT_ASSERT(from >= 0);
		BOLT_ASSERT(size <= this->size_);
		BOLT_ASSERT(from + size <= this->size_);

		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return DeviceArrayView(const_cast<Element *>(this->pointer_) + from, size);
	}

	DeviceArrayConstView<TElement> constSubview(IndexType from, SizeType size) const {
		return Predecessor::subview(from, size);
	}
};


template<typename TElement>
BOLT_DECL_HYBRID
DeviceArrayConstView<TElement>::DeviceArrayConstView(const DeviceArrayView<const Element> &view):
	pointer_(view.pointer()),
	size_(view.size())
{}

template<typename TElement>
BOLT_DECL_HYBRID
DeviceArrayConstView<TElement> &
DeviceArrayConstView<TElement>::operator=(const DeviceArrayView<Element> &view) {
	pointer_ = view.pointer();
	size_ = view.size();
	return *this;
}

template<typename TElement>
struct IsArrayView<DeviceArrayView<TElement>> : std::integral_constant<bool, true> {};

template<typename TElement>
struct IsArrayView<DeviceArrayConstView<TElement>> : std::integral_constant<bool, true> {};


template<typename TElement>
DeviceArrayConstView<const TElement>
makeDeviceArrayConstView(const TElement *buffer, int64_t size) {
	return DeviceArrayConstView<const TElement>(buffer, size);
}


template<typename TElement>
DeviceArrayView<TElement>
makeDeviceArrayView(TElement *buffer, int64_t size) {
	return DeviceArrayView<TElement>(buffer, size);
}


template<typename TElement>
DeviceArrayConstView<const TElement>
makeDeviceArrayConstView(const thrust::device_vector<TElement> &buffer) {
	return DeviceArrayConstView<const TElement>(thrust::raw_pointer_cast(buffer.data()), int64_t(buffer.size()));
}

template<typename TElement>
DeviceArrayView<TElement>
makeDeviceArrayView(thrust::device_vector<TElement> &buffer) {
	return DeviceArrayView<TElement>(thrust::raw_pointer_cast(buffer.data()), int64_t(buffer.size()));
}

template<typename TElement>
DeviceArrayView<TElement>
makeArrayView(thrust::device_vector<TElement> &buffer) {
	return DeviceArrayView<TElement>(thrust::raw_pointer_cast(buffer.data()), int64_t(buffer.size()));
}

template<typename TElement>
DeviceArrayConstView<TElement>
makeArrayConstView(const thrust::device_vector<const TElement> &buffer) {
	return DeviceArrayConstView<const TElement>(buffer.data().get(), int64_t(buffer.size()));
}

template<typename TElement>
DeviceArrayConstView<TElement>
makeArrayConstView(thrust::device_vector<TElement> &buffer) {
	return DeviceArrayConstView<TElement>(buffer.data().get(), int64_t(buffer.size()));
}



template<typename TElement>
auto view(thrust::device_vector<TElement> &buffer) {
	return DeviceArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
auto constView(thrust::device_vector<const TElement> &buffer) {
	return DeviceArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

#endif  // __CUDACC__

template<typename TElement>
class HostArrayView;

template<typename TElement>
class HostArrayConstView {
public:
	static const bool kIsDeviceView = false;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = 1;
	using SizeType = int64_t;
	using IndexType = int64_t;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element;
	using Policy = LongIndexViewPolicy;
	using TIndex = typename Policy::IndexType;

	HostArrayConstView() = default;

	HostArrayConstView(const Element *pointer, SizeType size) :
		pointer_(pointer),
		size_(size)
	{}

	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions
	HostArrayConstView(const HostArrayConstView<Element> &view) :
		pointer_(view.pointer()),
		size_(view.size())
	{}

	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions
	HostArrayConstView(const HostArrayConstView<const Element> &view):
		pointer_(view.pointer()),
		size_(view.size())
	{}

	// NOLINTNEXTLINE(google-explicit-constructor) -- allow implicit conversions
	HostArrayConstView(const HostArrayView<Element> &view);

	HostArrayConstView<TElement> &operator=(const HostArrayConstView<Element> &view) {
		pointer_ = view.pointer_;
		size_ = view.size_;
		return *this;
	}

	HostArrayConstView<TElement> &operator=(const HostArrayView<Element> &view);

	HostArrayConstView<TElement> &operator=(const HostArrayConstView<const Element> &view) {
		pointer_ = view.pointer_;
		size_ = view.size_;
		return *this;
	}

	const Element *begin() const {
		return this->pointer_;
	}

	const Element *end() const {
		return this->pointer_ + this->size_;
	}


	SizeType size()  const {
		return size_;
	}

	SizeType elementCount()  const {
		return size_;
	}

	bool empty() const {
		return size_ == 0;
	}

	const Element *pointer() const {
		return pointer_;
	}

	AccessType operator[](IndexType index) const {
		return pointer_[index];
	}

	/// Creates view for part of this view.
	HostArrayConstView<TElement> subview(IndexType from, SizeType size) const {
		BOLT_ASSERT(from >= 0);
		BOLT_ASSERT(size <= size_);
		BOLT_ASSERT(from + size <= size_);

		return HostArrayConstView(pointer_ + from, size);
	}

	int strides() const {
		return 1;
	}

protected:
	const Element *pointer_ = nullptr;
	SizeType size_ = 0;
};


template<typename TElement>
class HostArrayView: public HostArrayConstView<TElement> {
public:
	static const bool kIsDeviceView = false;
	static const bool kIsHostView = true;
	static const bool kIsMemoryBased = true;
	static const int kDimension = 1;
	using SizeType = int64_t;
	using IndexType = int64_t;
	using Predecessor = HostArrayConstView<TElement>;
	using Element = typename std::remove_const<TElement>::type;
	using AccessType = Element &;
	using Policy = LongIndexViewPolicy;
	using TIndex = typename Policy::IndexType;

	HostArrayView() = default;

	HostArrayView(TElement *pointer, SizeType size) :
		Predecessor(pointer, size)
	{}

	HostArrayView(const HostArrayView &) = default;
	HostArrayView(HostArrayView &&) = default;
	~HostArrayView() = default;

	HostArrayView &operator=(const HostArrayView &) = default;
	HostArrayView &operator=(HostArrayView &&) = default;


	AccessType operator[](IndexType index) const {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_)[index];
	}

	Element *pointer() {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_);
	}

	const Element *pointer() const {
		return this->pointer_;
	}

	Element *begin() {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_);
	}

	Element *end() {
		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return const_cast<Element *>(this->pointer_ + this->size_);
	}

	/// Creates view for part of this view.
	HostArrayView<TElement> subview(IndexType from, SizeType size) const {
		BOLT_ASSERT(from >= 0);
		BOLT_ASSERT(size <= this->size_);
		BOLT_ASSERT(from + size <= this->size_);

		// NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
		return HostArrayView(const_cast<Element *>(this->pointer_) + from, size);
	}

	HostArrayConstView<TElement> constSubview(IndexType from, SizeType size) const {
		return Predecessor::Subview(from, size);
	}

	/*template<class Archive>
	void serialize(Archive & ar, const unsigned int file_version){
		if (Archive::is_loading::value)
		{
			SizeType load_size;
			ar & boost::serialization::make_nvp("size", load_size);
			if (load_size != this->size_) {
				BOLT_THROW(BoltError());
			}
		} else {
			ar & boost::serialization::make_nvp("size", this->size_);
		}
		ar & boost::serialization::make_array<Element>(this->pointer_, this->size_);
	}*/
};

template<typename TElement>
HostArrayConstView<TElement>::HostArrayConstView(const HostArrayView<Element> &view):
	pointer_(view.pointer()),
	size_(view.size())
{}


template<typename TElement>
HostArrayConstView<TElement> &
HostArrayConstView<TElement>::operator=(const HostArrayView<Element> &view) {
	pointer_ = view.pointer_;
	size_ = view.size_;
	return *this;
}


template<typename TElement>
struct IsArrayView<HostArrayView<TElement>> : std::integral_constant<bool, true> {};

template<typename TElement>
struct IsArrayView<HostArrayConstView<TElement>> : std::integral_constant<bool, true> {};

template<typename TElement>
HostArrayConstView<const TElement>
makeHostArrayConstView(const TElement *buffer, int64_t size) {
	return HostArrayConstView<const TElement>(buffer, size);
}


template<typename TElement>
HostArrayView<TElement>
makeHostArrayView(TElement *buffer, int64_t size) {
	return HostArrayView<TElement>(buffer, size);
}

template<typename TElement>
HostArrayConstView<const TElement>
makeHostArrayConstView(const std::vector<TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<const TElement>
makeHostArrayConstView(const thrust::host_vector<TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
makeHostArrayView(std::vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
makeArrayView(std::vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<const TElement>
makeArrayConstView(std::vector<TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
makeArrayConstView(std::vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
makeHostArrayView(thrust::host_vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
makeArrayView(thrust::host_vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
makeArrayConstView(thrust::host_vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
makeArrayConstView(thrust::host_vector<TElement> &buffer) {
	return HostArrayConstView<TElement>(buffer.data(), int64_t(buffer.size()));
}

// -------------------------------------------

template<typename TElement>
HostArrayView<TElement>
view(std::vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<const TElement>
constView(std::vector<TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
constView(std::vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayView<TElement>
view(thrust::host_vector<TElement> &buffer) {
	return HostArrayView<TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
constView(thrust::host_vector<const TElement> &buffer) {
	return HostArrayConstView<const TElement>(buffer.data(), int64_t(buffer.size()));
}

template<typename TElement>
HostArrayConstView<TElement>
constView(thrust::host_vector<TElement> &buffer) {
	return HostArrayConstView<TElement>(buffer.data(), int64_t(buffer.size()));
}



//TODO(johny) - size instead of last
BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
TView arraySubview(TView view, int first, int last) {
	auto pointer = view.pointer() + first;
	return TView(pointer, last - first);
}

template<typename TElement>
HostArrayConstView<TElement>
subview(HostArrayConstView<TElement> view, int first, int last) {
	return arraySubview(view, first, last);
}

template<typename TElement>
HostArrayView<TElement>
subview(HostArrayView<TElement> view, int first, int last) {
	return arraySubview(view, first, last);
}

#if defined(__CUDACC__)

template<typename TElement>
DeviceArrayConstView<TElement>
subview(DeviceArrayConstView<TElement> view, int first, int last) {
	return arraySubview(view, first, last);
}

template<typename TElement>
DeviceArrayView<TElement>
subview(DeviceArrayView<TElement> view, int first, int last) {
	return arraySubview(view, first, last);
}
#endif  // __CUDACC__

}  // namespace bolt
