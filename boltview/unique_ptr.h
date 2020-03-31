// Copyright 2017 Eyen SE
// Author: Lukas Marsalek, lukas.marsalek@eyen.eu

#pragma once

#include <type_traits>
#include <algorithm>
#include <boltview/cuda_utils.h>

namespace bolt {
namespace device {

/// Functor for deleting device pointers on host using cudaFree() and on device using free()
template<class TType>
struct HybridDeleter {
	/// Default constructor, explicitly declared with constexpr
	constexpr HybridDeleter() = default;

	/// The function call operator does the actual delete using cudaFree() or free()
	BOLT_DECL_HYBRID
	void operator()(TType* ptr) const{
		if(ptr){
			#ifdef __CUDA_ARCH__
				free(ptr);
			#else
				try {
					BOLT_CHECK(cudaFree(ptr));
				} catch (CudaError &e) {
					BOLT_ERROR_FORMAT("Device image deallocation failure.: %1%", boost::diagnostic_information(e));
				}
			#endif
		}
	}

	/// All casts to pointers other than \tparam TType are explicitly forbiden
	/// This prevents deleting other pointers that the one passed as \tparam TType
	template<class U>
	void operator()(U*) const = delete;
};

/// Functor encapsulating device memory allocation from host including
/// failure check
template<class TType>
struct HostAllocator {
	/// The function call operator doing the actuall allocation using cudaMalloc()
	TType* operator()(int64_t size) const {
		TType* newPtr = nullptr;
		BOLT_ASSERT(size > 0 && "Size for allocation must be bigger than 0.");
		try{
			BOLT_CHECK(cudaMalloc(&newPtr, size));
		} catch (CudaError &e) {
			D_MEM_OUT(size);
			throw e;
		}
		return newPtr;
	}
};

/// Functor encapsulating device memory allocation from device
template<class TType>
struct DeviceAllocator {
	/// The function call operator doing allocation on device using malloc()
	BOLT_DECL_DEVICE
	TType* operator()(int64_t size) const {
		return reinterpret_cast<TType*>(malloc(size));
	}
};
/// unique_ptr equivalent for device pointers. Its primary purpose is to enable
/// transparent resource management on the same level as host unique_ptr.
/// First incentive for its introduction was to avoid move constructors in
/// %DeviceImage class while still allowing it to be stored in STL containers
/// Naming note: It intentionally follows the naming scheme of unique_ptr instead
/// of BoltView's naming conventions to emphasize the equivalent meaning and usage
template <class TType, class TDeleter>
class unique_ptr_with_deleter
{
public:
	using Element_type = TType;
	using Pointer_type = Element_type*;
	using Deleter_type = TDeleter;
	using Self_type = unique_ptr_with_deleter<TType, TDeleter>;

	/// Default constructor that initializes to no ownership. Ownership is then
	/// acquired either through unique_ptr(Self_type&& u), unique_ptr(Pointer_type p)
	/// or a call to unique_ptr::reset(Pointer_type p).
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	unique_ptr_with_deleter() :
		deleter_(Deleter_type()),
		ptr_(nullptr)
	{}

	/// Constructor from pointer. This is the default way to assume ownership
	/// of a pointer. Note that this constructor requires you to explicitly
	/// specify the deleter to be used. The reason is that we cannot infer
	/// from just the pointer type whether it was allocated on host or device.
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	explicit unique_ptr_with_deleter(Pointer_type p) :
		deleter_(Deleter_type()),
		ptr_(p)
	{}

	/// Move constructor. Takes over ownership of the right-hand side pointer.
	/// This is the default way to change unique_ptr-owned pointer ownership
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	unique_ptr_with_deleter(Self_type&& rhs) :
		deleter_(rhs.deleter_),
		ptr_(rhs.ptr_)
	{
		rhs.ptr_ = nullptr;
	}

	/// Specific constructor designed for nullptr. In behaviour equivalent
	/// to the default constructor
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	explicit unique_ptr_with_deleter(std::nullptr_t) :
		unique_ptr_with_deleter()
	{}

	/// Constructor from pointer - default way to assume ownership of a pointer
	/// This overload additionally specifies custom deleter
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	unique_ptr_with_deleter(Pointer_type p, Deleter_type del) :
		deleter_(del),
		ptr_(p)
	{}

	/// Destructor
	BOLT_DECL_HYBRID
	~unique_ptr_with_deleter(){
		reset();
	}

	/// Copy assignments are disabled as %unique_ptr is non-copyable.
	Self_type& operator=(const Self_type& other) = delete;

	/// Copy assignment from another pointer type is also disabled
	template <class TOtherType, class TOtherDeleter>
	Self_type& operator=(unique_ptr_with_deleter<TOtherType, TOtherDeleter>&& rhs) = delete;

	/// Copy assignment for nullptr is enabled, as it does not violate the
	/// ownership paradigm and is equivalent to reset()
	BOLT_DECL_HYBRID
	Self_type& operator=(std::nullptr_t) {
		reset(nullptr);
	}

	/// Move assignment is allowed as it represents transfer of ownership
	BOLT_DECL_HYBRID
	Self_type& operator=(Self_type&& rhs){
		reset(rhs.ptr_);
		deleter_ = rhs.deleter_;
		rhs.ptr_ = nullptr;
		return *this;
	}

	/// Dereference operator
	BOLT_DECL_HYBRID
	Element_type& operator*() const {
		return *ptr_;
	}

	/// Member access operator
	BOLT_DECL_HYBRID
	Pointer_type operator->() const {
		return ptr_;
	}

	/// Returns the owned raw pointer, while still keeeping its ownership
	/// (unlike release()).
	BOLT_DECL_HYBRID
	Pointer_type get() const {
		return ptr_;
	}

	/// Returns mutable associated deleter functor
	BOLT_DECL_HYBRID
	Deleter_type& get_deleter() {
		return deleter_;
	}

	/// Return immutable associated deleter functor
	BOLT_DECL_HYBRID
	const Deleter_type& get_deleter() const {
		return deleter_;
	}

	/// nullptr check operator.
	/// Returns true, if owned pointer is != nullptr
	BOLT_DECL_HYBRID
	explicit operator bool() const {
		return ptr_ != nullptr;
	}

	/// Return the raw pointed-to pointer to the caller and reset own state
	/// to point to nothing.
	BOLT_DECL_HYBRID
	Pointer_type release() {
		Pointer_type tmpPtr = ptr_;
		ptr_ = nullptr;
		return tmpPtr;
	}

	/// Resets the state of the %unique_ptr to own no pointer
	/// The data pointed to so far, if any, are DELETED
	BOLT_DECL_HYBRID
	void reset() {
		reset(nullptr);
	}

	/// This method deletes the pointed-to data and sets the unique_ptr
	/// to own \param p
	/// It is a convenience function to avoid chained reset, delete and construct.
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	void reset(Pointer_type p) {
		Pointer_type tmpPtr = ptr_;
		ptr_ = p;
		if(tmpPtr){
			deleter_(tmpPtr);
		}
	}

	/// Specific reset for nullptr, which is the only other pointer type
	// than ::Element_type*, for which reset makes sense
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	void reset(std::nullptr_t) {
		Pointer_type tmpPtr = release();
		if(tmpPtr){
			deleter_(tmpPtr);
		}
	}

	/// Explicitly forbid to reset to a pointer to another class than the one
	/// given as \tparam TType
	template<class TOtherType>
	void reset(TOtherType* p) = delete;

	/// Swap two %unique_ptr pointing to the same type with the same deleter_
	void swap(Self_type& rhs) {
		std::swap(ptr_, rhs.ptr_);
		std::swap(deleter_, rhs.deleter_);
	}

protected:
	Pointer_type ptr_;
	Deleter_type deleter_;
};

/// Final declaration of device::unique_ptr using %HybridDeleter
template<class TType>
using unique_ptr = unique_ptr_with_deleter<TType, HybridDeleter<TType>>;

/// Equivalent of std::make_unique for device::unique_ptr
template<class TType>
BOLT_DECL_HYBRID
unique_ptr<TType>
make_unique(){
	#ifdef __CUDA_ARCH__
		return unique_ptr<TType>(DeviceAllocator<TType>()(sizeof(TType)));
	#else
		return unique_ptr<TType>(HostAllocator<TType>()(sizeof(TType)));
	#endif
}

/// Equivalent of std::make_unique for array allocation. Unlike the std variant,
/// this method is callable also on non-array types, since the deletion mechanism
/// is array-independent. Otherwise it is equivalnt to std in the sense that the
/// \param numElements represent number of elements to allocate, NOT a byte size of the
/// allocation.
template<class TType>
BOLT_DECL_HYBRID
unique_ptr<TType>
make_unique(const int64_t numElements){
	#ifdef __CUDA_ARCH__
		return unique_ptr<TType>(DeviceAllocator<TType>()(sizeof(TType) * numElements));
	#else
		return unique_ptr<TType>(HostAllocator<TType>()(sizeof(TType) * numElements));
	#endif
}
}  // namespace device
}  // namespace bolt
