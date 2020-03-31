#pragma once

#include <type_traits>

#include <boltview/cuda_defines.h>

namespace bolt {

#ifdef __CUDACC__
template<typename TItem, int tSize, bool tCallConstructors = false>
class SharedMemoryStaticArray {
public:
	BOLT_DECL_DEVICE
	SharedMemoryStaticArray()
	{
		callConstructors(std::integral_constant<bool, tCallConstructors>{});
		__syncthreads();
	}

	BOLT_DECL_DEVICE
	TItem &operator[](int index) {
		return *reinterpret_cast<TItem *>(Data() + index * sizeof(TItem));
	}

	BOLT_DECL_DEVICE
	const TItem &operator[](int index)const {
		return *reinterpret_cast<const TItem *>(Data() + index * sizeof(TItem));
	}

private:

	BOLT_DECL_DEVICE
	int8_t *Data() {
		static __shared__ int8_t data[sizeof(TItem) * tSize];
		return data;
	}

	BOLT_DECL_DEVICE
	void callConstructors(std::true_type) {
		int index = threadOrderFromIndex();
		while(index < tCallConstructors) {
			index += currentBlockSize();
			TItem *address = &((*this)[index]);
			new(address) TItem();
		}
	}

	BOLT_DECL_DEVICE
	void callConstructors(std::false_type) {

	}

};

#endif  // __CUDACC__
} // namespace bolt
