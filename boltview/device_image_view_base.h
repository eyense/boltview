// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once



#include <boltview/cuda_utils.h>
#include <boltview/cuda_defines.h>
#include <boltview/region.h>
#include <boltview/view_policy.h>

namespace bolt {


BOLT_DECL_HYBRID
inline bool foldAnd(const Vector<bool, 2> &values) {
	return values[0] && values[1];
}

BOLT_DECL_HYBRID
inline bool foldAnd(const Vector<bool, 3> &values) {
	return values[0] && values[1] && values[2];
}
/// \addtogroup Views
/// @{

#if defined(__CUDACC__)
/// Base class for various image views.
/// View concept should provide description (size) of the N-D image and
/// provide access to its elements (pixels, voxels) through operator[].
/// Provides methods for size and coordinate queries.
template<int tDimension, typename TPolicy = DefaultViewPolicy>
class DeviceImageViewBase {
public:
	static const bool kIsHostView = false;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = false;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using IndexType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	static const int kDimension = tDimension;

	BOLT_DECL_HYBRID
	explicit DeviceImageViewBase(SizeType size) :
		size_(size)
	{}

	/// \return N-dimensional size.
	BOLT_DECL_HYBRID
	SizeType size() const {
		return size_;
	}

	/// \return Total number of accessible elements
	BOLT_DECL_HYBRID
	TIndex elementCount() const {
		return product(size_);
	}

	/// Checks if provided index points inside current view.
	BOLT_DECL_HYBRID
	bool isIndexInside(const IndexType &index) const {
		#ifdef __CUDA_ARCH__
			// optimized for memory dependencies - interval checks can be done in parallel
			Vector<bool, tDimension> inside;
			#pragma unroll
			for (int i = 0; i < tDimension; ++i) {
				inside[i] = index[i] >= 0 && index[i] < size_[i];
			}
			return foldAnd(inside);
		#else
			bool inside = true;
			for (int i = 0; inside && i < tDimension; ++i) {
				inside = inside && index[i] >= 0 && index[i] < size_[i];
			}
			return inside;
		#endif //__CUDA_ARCH__
	}

	BOLT_DECL_HYBRID
	Region<tDimension, TIndex> getRegion() const {
		return Region<tDimension, TIndex>{ IndexType(), size() };
	}
protected:
	// TODO(johny) - view with offsets
	SizeType size_;
};
// #else  // TODO(johny) check if feasible compile time error report
//
// template<int tDimension, typename TPolicy = DefaultViewPolicy>
// class DeviceImageViewBase {
// public:
// 	static_assert(false, "DeviceImageViewBase can be used only in code compiled by nvcc.");
// };
#endif  // !defined(__CUDACC__)

template<int tDimension, typename TPolicy = DefaultViewPolicy>
class HybridImageViewBase {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = true;
	static const bool kIsMemoryBased = false;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using IndexType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	static const int kDimension = tDimension;

	BOLT_DECL_HYBRID
	explicit HybridImageViewBase(SizeType size) :
		size_(size)
	{}

	/// \return N-dimensional size.
	BOLT_DECL_HYBRID
	SizeType size() const {
		return size_;
	}

	/// \return Total number of accessible elements
	BOLT_DECL_HYBRID
	TIndex elementCount() const {
		return product(size_);
	}

	/// Checks if provided index points inside current view.
	BOLT_DECL_HYBRID
	bool isIndexInside(const IndexType &index) const {
		bool inside = true;
		for (int i = 0; inside && i < tDimension; ++i) {
			inside = inside && index[i] >= 0 && index[i] < size_[i];
		}
		return inside;
	}

	BOLT_DECL_HYBRID
	Region<tDimension, TIndex> getRegion() const {
		return Region<tDimension, TIndex>{ IndexType(), size() };
	}
protected:
	// TODO(johny) - view with offsets
	SizeType size_;
};

/// @}


}  // namespace bolt
