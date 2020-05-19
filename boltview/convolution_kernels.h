// Copyright 2017 Eyen SE
// Author: Jan Cerveny jan.cerveny@eyen.se

#pragma once

#include <algorithm>
#include <memory>

namespace bolt {


/// Test kernel with all weights constant
template<typename TType, int tDimension>
class ConstKernel {
public:
	static const bool kIsDynamicallyAllocated = false;
	static const bool kIsHostKernel = true;
	static const bool kIsDeviceKernel = true;
	static const int kDimension = tDimension;


	BOLT_DECL_HYBRID
	ConstKernel(TType val, Vector<int, tDimension> size, Vector<int, tDimension> center):
		value_(val), size_(size), center_(center)
	{}

	BOLT_DECL_HYBRID
	TType operator[](Vector<int, tDimension>  /*index*/) const{
		return value_;
	}

	BOLT_DECL_HYBRID
	Vector<int, kDimension> size() const{
		return size_;
	}

	BOLT_DECL_HYBRID
	Vector<int, kDimension> center() const{
		return center_;
	}

private:
	TType value_;
	Vector<int, tDimension> size_;
	Vector<int, tDimension> center_;
};


/// Base class for holding convolution kernel and its proportions
template<typename TValue, int tDimension>
class DynamicKernelBase {
public:
	static const bool kIsDynamicallyAllocated = true;
	static const int kDimension = tDimension;
	using ElementType = TValue;

	BOLT_DECL_HYBRID
	DynamicKernelBase(Vector<int, tDimension> size, Vector<int, tDimension> center):
		size_(size), center_(center)
	{}

	BOLT_DECL_HYBRID
	DynamicKernelBase(const DynamicKernelBase &) = default;

	BOLT_DECL_HYBRID
	DynamicKernelBase(DynamicKernelBase &&) = default;

	BOLT_DECL_HYBRID
	DynamicKernelBase &operator=(const DynamicKernelBase &) = default;

	BOLT_DECL_HYBRID
	DynamicKernelBase &operator=(DynamicKernelBase &&) = default;

	BOLT_DECL_HYBRID
	TValue operator[](Vector<int, tDimension> index) const{
		return kernel_[kernelIndex(*this, index)];
	}

	BOLT_DECL_HYBRID
	Vector<int, kDimension> size() const{
		return size_;
	}

	BOLT_DECL_HYBRID
	Vector<int, kDimension> center() const{
		return center_;
	}

	BOLT_DECL_HYBRID
	TValue * pointer() const{
		return kernel_;
	}

protected:
	TValue * kernel_;

private:
	Vector<int, tDimension> size_;
	Vector<int, tDimension> center_;
};

/*template<typename TValue, int tDimension>
struct DeviceViewKernel : public DynamicKernelBase<TValue, tDimension>{
	static const bool kIsHostKernel = false;
	static const bool kIsDeviceKernel = true;
	using Predecessor = DynamicKernelBase<TValue, tDimension>;

	BOLT_DECL_HYBRID
	DeviceViewKernel(DeviceImageView<TValue, 3> view):
	DynamicKernelBase<TValue, tDimension>(view.size(), bolt::Div(view.size(), 2))
	{
		this->kernel_ = view.pointer();
	}

};*/

/// Kernel stored on host
template<typename TValue, int tDimension>
struct DynamicHostKernel : public DynamicKernelBase<TValue, tDimension>{
	static const bool kIsHostKernel = true;
	static const bool kIsDeviceKernel = false;
	using Predecessor = DynamicKernelBase<TValue, tDimension>;

	/// \param size size of kernel
	/// \param center center of kernel
	/// \param ptr Pointer to values to be copied to this kernel
	DynamicHostKernel(Vector<int, tDimension> size, Vector<int, tDimension> center, TValue * ptr):
		DynamicKernelBase<TValue, tDimension>(size, center), ptr_(new TValue[product(size)])
	{
		this->kernel_ = ptr_.get();
		std::copy(ptr, ptr + product(size), this->kernel_);
	}

	DynamicHostKernel(const DynamicHostKernel &) = delete;
	DynamicHostKernel(DynamicHostKernel &&) = default;

	DynamicHostKernel &operator=(const DynamicHostKernel &) = delete;
	DynamicHostKernel &operator=(DynamicHostKernel &&) = default;

private:
	std::unique_ptr<TValue[]> ptr_;
};

#ifdef BOLT_USE_UNIFIED_MEMORY

/// Kernel stored in unified memory
template<typename TValue, int tDimension>
struct DynamicUnifiedKernel : public DynamicKernelBase<TValue, tDimension>{
	static const bool kIsHostKernel = true;
	static const bool kIsDeviceKernel = true;
	using Predecessor = DynamicKernelBase<TValue, tDimension>;

	/// \param size size of kernel
	/// \param center center of kernel
	/// \param ptr Pointer to values to be copied to this kernel
	DynamicUnifiedKernel(Vector<int, tDimension> size, Vector<int, tDimension> center, TValue * ptr):
		DynamicKernelBase<TValue, tDimension>(size, center)
	{
		int num_elements = product(size);
		BOLT_CHECK(cudaMallocManaged(&(this->kernel_), sizeof(TValue) * num_elements));
		cudaDeviceSynchronize();
		memcpy(this->kernel_, ptr, sizeof(*ptr) * num_elements);
	}

	DynamicUnifiedKernel(const DynamicUnifiedKernel &) = delete;

	DynamicUnifiedKernel(DynamicUnifiedKernel && other):
		DynamicKernelBase<TValue, tDimension>(other.size(), other.center())
	{
		this->kernel_ = other.kernel_;
		other.kernel_ = nullptr;
	}

	~DynamicUnifiedKernel(){
		if (this->kernel_) {
			try {
				cudaDeviceSynchronize();
				BOLT_CHECK(cudaFree(this->kernel_));
			} catch (CudaError &e) {
				BOLT_ERROR_FORMAT("Unified image deallocation failure.: %1%", boost::diagnostic_information(e));
			}
		}
	}
};

#endif // BOLT_USE_UNIFIED_MEMORY


/// Kernel stored on device
template<typename TValue, int tDimension>
struct DynamicDeviceKernel : public DynamicKernelBase<TValue, tDimension>{
	static const bool kIsHostKernel = false;
	static const bool kIsDeviceKernel = true;
	using Predecessor = DynamicKernelBase<TValue, tDimension>;

	/// \param size size of kernel
	/// \param center center of kernel
	/// \param ptr Pointer to values to be copied to this kernel
	DynamicDeviceKernel(Vector<int, tDimension> size, Vector<int, tDimension> center, TValue * ptr):
		DynamicKernelBase<TValue, tDimension>(size, center)
	{
		int num_elements = product(size);
		BOLT_CHECK(cudaMalloc(&(this->kernel_), sizeof(*ptr) * num_elements));
		cudaMemcpy(this->kernel_, ptr, sizeof(*ptr) * num_elements, cudaMemcpyHostToDevice);
	}

	DynamicDeviceKernel(const DynamicHostKernel<TValue, tDimension> &k):
		DynamicDeviceKernel(k.size(), k.center(), k.pointer())
	{
	}

	DynamicDeviceKernel(const DynamicDeviceKernel &) = delete;

	~DynamicDeviceKernel(){
		if (this->kernel_) {
			try {
				BOLT_CHECK(cudaFree(this->kernel_));
			} catch (CudaError &e) {
				BOLT_ERROR_FORMAT("Device image deallocation failure.: %1%", boost::diagnostic_information(e));
			}
		}
	}
};

#ifdef BOLT_USE_UNIFIED_MEMORY

/// Holds separable kernel
template<typename TType, int tDimension>
class SeparableKernel{
public:
	static const int kDimension = tDimension;

	SeparableKernel(TType * array, Vector<int, tDimension> size, Vector<int, tDimension> center):
		array_(new TType[sum(size)]), size_(size), center_(center)
	{
			memcpy(array_.get(), array, sum(size) * sizeof(TType));
	}

	/// \return 1D kernel stored in unified memory with properly set size nad center
	DynamicUnifiedKernel<TType, tDimension> get(int axis) const{
		auto kernel_size = Vector<int, tDimension>::fill(1);
		auto kernel_center = Vector<int, tDimension>::fill(0);

		kernel_size[axis] = size_[axis];
		kernel_center[axis] = center_[axis];

		int size = 0;
		for(int i = 0; i < axis; ++i){
			size += size_[i];
		}

		return DynamicUnifiedKernel<TType, tDimension>(kernel_size, kernel_center, array_.get() + size);
	}

private:
	Vector<int, tDimension> size_;
	Vector<int, tDimension> center_;
	std::unique_ptr<TType[]> array_;
};

#endif // BOLT_USE_UNIFIED_MEMORY

}  // namespace bolt
