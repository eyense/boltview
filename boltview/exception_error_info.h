#pragma once

#include <boost/exception/all.hpp>

#include <boltview/math/vector.h>
#include <boltview/region.h>


namespace bolt {

/// \addtogroup Utilities
/// @{

/// \addtogroup ExceptionErrorInfo
/// @{

template<typename TSize>
struct SizePair: std::pair<TSize, TSize> {
	using std::pair<TSize, TSize>::pair;
};

template<typename TSize>
std::ostream &operator<<(std::ostream &out, const SizePair<TSize> &size_pair) {
	return out << "{ " << size_pair.first << ", " << size_pair.second << " }";
}


#define DEFINE_VIEW_PAIR_SIZES_EXN(type_) /*NOLINT*/ \
using ViewPairSizesErrorInfo ## type_ = boost::error_info<struct tag_index_view_pair_sizes_ ## type_, SizePair<type_>>; /*NOLINT*/\
inline ViewPairSizesErrorInfo ## type_ getViewPairSizesErrorInfo(const type_ &first, const type_ &second) { \
	return ViewPairSizesErrorInfo ## type_(SizePair<type_>(first, second)); \
}

DEFINE_VIEW_PAIR_SIZES_EXN(int64_t)
// DEFINE_VIEW_PAIR_SIZES_EXN(Int1)
DEFINE_VIEW_PAIR_SIZES_EXN(Int2)
DEFINE_VIEW_PAIR_SIZES_EXN(Int3)
// DEFINE_VIEW_PAIR_SIZES_EXN(LongInt1)
DEFINE_VIEW_PAIR_SIZES_EXN(LongInt2)
DEFINE_VIEW_PAIR_SIZES_EXN(LongInt3)

using OriginalRegion1DErrorInfo = boost::error_info<struct tag_original_region_1d, Region<1>>;
using OriginalRegion2DErrorInfo = boost::error_info<struct tag_original_region_2d, Region<2>>;
using OriginalRegion3DErrorInfo = boost::error_info<struct tag_original_region_3d, Region<3>>;
using OriginalLongIntRegion1DErrorInfo = boost::error_info<struct tag_original_region_1d, Region<1, int64_t>>;
using OriginalLongIntRegion2DErrorInfo = boost::error_info<struct tag_original_region_2d, Region<2, int64_t>>;
using OriginalLongIntRegion3DErrorInfo = boost::error_info<struct tag_original_region_3d, Region<3, int64_t>>;

inline OriginalRegion1DErrorInfo getOriginalRegionErrorInfo(const Region<1> &region) {
	return OriginalRegion1DErrorInfo(region);
}

inline OriginalRegion2DErrorInfo getOriginalRegionErrorInfo(const Region<2> &region) {
	return OriginalRegion2DErrorInfo(region);
}

inline OriginalRegion3DErrorInfo getOriginalRegionErrorInfo(const Region<3> &region) {
	return OriginalRegion3DErrorInfo(region);
}

inline OriginalLongIntRegion1DErrorInfo getOriginalRegionErrorInfo(const Region<1, int64_t> &region) {
	return OriginalLongIntRegion1DErrorInfo(region);
}

inline OriginalLongIntRegion2DErrorInfo getOriginalRegionErrorInfo(const Region<2, int64_t> &region) {
	return OriginalLongIntRegion2DErrorInfo(region);
}

inline OriginalLongIntRegion3DErrorInfo getOriginalRegionErrorInfo(const Region<3, int64_t> &region) {
	return OriginalLongIntRegion3DErrorInfo(region);
}

using WrongRegion1DErrorInfo = boost::error_info<struct tag_wrong_region_1d, Region<1>>;
using WrongRegion2DErrorInfo = boost::error_info<struct tag_wrong_region_2d, Region<2>>;
using WrongRegion3DErrorInfo = boost::error_info<struct tag_wrong_region_3d, Region<3>>;
using WrongLongIntRegion1DErrorInfo = boost::error_info<struct tag_wrong_region_1d, Region<1, int64_t>>;
using WrongLongIntRegion2DErrorInfo = boost::error_info<struct tag_wrong_region_2d, Region<2, int64_t>>;
using WrongLongIntRegion3DErrorInfo = boost::error_info<struct tag_wrong_region_3d, Region<3, int64_t>>;

inline WrongRegion1DErrorInfo getWrongRegionErrorInfo(const Region<1> &region) {
	return WrongRegion1DErrorInfo(region);
}

inline WrongRegion2DErrorInfo getWrongRegionErrorInfo(const Region<2> &region) {
	return WrongRegion2DErrorInfo(region);
}

inline WrongRegion3DErrorInfo getWrongRegionErrorInfo(const Region<3> &region) {
	return WrongRegion3DErrorInfo(region);
}

inline WrongLongIntRegion2DErrorInfo getWrongRegionErrorInfo(const Region<2, int64_t> &region) {
	return WrongLongIntRegion2DErrorInfo(region);
}

inline WrongLongIntRegion3DErrorInfo getWrongRegionErrorInfo(const Region<3, int64_t> &region) {
	return WrongLongIntRegion3DErrorInfo(region);
}

// First the slice number, then slicing dimension
using WrongSliceErrorInfo = boost::error_info<struct tag_wrong_slice, Int2>;


using Size1DErrorInfo = boost::error_info<struct tag_size_1d, int>;
using Size2DErrorInfo = boost::error_info<struct tag_size_2d, Int2>;
using Size3DErrorInfo = boost::error_info<struct tag_size_3d, Int3>;

inline Size1DErrorInfo getSizeErrorInfo(int size) {
	return Size1DErrorInfo(size);
}

inline Size2DErrorInfo getSizeErrorInfo(Int2 size) {
	return Size2DErrorInfo(size);
}

inline Size3DErrorInfo getSizeErrorInfo(Int3 size) {
	return Size3DErrorInfo(size);
}

/// @}
/// @}

}  // namespace bolt
