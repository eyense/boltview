// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <boltview/math/vector.h>
#include <boost/serialization/nvp.hpp>

namespace bolt {

template<int tDimension, typename TIndex = int>
struct Region
{
	Vector<TIndex, tDimension> corner;
	Vector<TIndex, tDimension> size;
};

template<int tDimension, typename TIndex = int>
BOLT_DECL_HYBRID
Region<tDimension, TIndex> createRegion(const Vector<TIndex, tDimension> &corner, const Vector<TIndex, tDimension> &size) {
	return Region<tDimension, TIndex>{ corner, size };
}

template<int tDimension, typename TIndex = int>
BOLT_DECL_HYBRID
Vector<TIndex, tDimension> clampToRegion(const Region<tDimension, TIndex> &region, const Vector<TIndex, tDimension> &coords) {
	return Min(Max(coords, region.corner), region.corner + region.size - Vector<TIndex, tDimension>::Fill(1));
}

/*template<int tDimension, typename TIndex = int>
Region<tDimension, TIndex> &operator+=(Region<tDimension, TIndex> &region, const Vector<TIndex, tDimension> &offset) {
	region.corner += offset;
	return region;
}*/


/*template<int tDimension, typename TIndex>
BOLT_DECL_HYBRID
bool
IsInsideRegion(const Vector<TIndex, tDimension> &aSize, const simple_vector<TIndex, tDimension> &aCoords)
{
	return aCoords >= simple_vector<TIndex, tDimension>() && aCoords < aSize;
}*/

template<int tDimension, typename TIndex = int>
std::ostream &operator<<(std::ostream &stream, const Region<tDimension, TIndex> &region) {
	return stream << '{' << region.corner << "; " << region.size << '}';
}

}  // namespace bolt

namespace boost {
namespace serialization {

template<class TArchive, int tDimension, typename TIndex = int>
void serialize(TArchive &ar, bolt::Region<tDimension, TIndex> &region, const unsigned int  /*version*/) {
	ar & boost::serialization::make_nvp("corner", region.corner);
	ar & boost::serialization::make_nvp("size", region.size);
}

} // namespace serialization
} // namespace boost



