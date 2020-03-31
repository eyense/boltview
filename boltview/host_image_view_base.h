// Copyright 2016 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/region.h>
#include <boltview/view_policy.h>

namespace bolt {

/// \addtogroup Views
/// @{

/// Base class for various image views.
/// View concept should provide description (size) of the N-D image and
/// provide access to its elements (pixels, voxels) through operator[].
/// Provides methods for size and coordinate queries.
template<int tDimension, typename TPolicy = DefaultViewPolicy>
class HostImageViewBase {
public:
	static const bool kIsHostView = true;
	static const bool kIsDeviceView = false;
	static const bool kIsMemoryBased = false;
	using Policy = TPolicy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	using IndexType = typename VectorTraits<Vector<TIndex, tDimension>>::type;
	static const int kDimension = tDimension;

	explicit HostImageViewBase(SizeType size) :
		size_(size)
	{}

	/// \return N-dimensional size.
	SizeType size() const {
		return size_;
	}

	/// \return Total number of accessible elements
	TIndex elementCount() const {
		return product(size());
	}

	/// Checks if provided index points inside current view.
	bool isIndexInside(const IndexType &index) const {
		bool inside = true;
		for (int i = 0; inside && i < tDimension; ++i) {
			inside = inside && index[i] >= 0 && index[i] < size()[i];
		}
		return inside;
	}

	Region<tDimension, TIndex> getRegion() const {
		return Region<tDimension, TIndex>{ IndexType(), size() };
	}

protected:
	// TODO(johny) - view with offsets
	SizeType size_;
};

/// @}


}  // namespace bolt
