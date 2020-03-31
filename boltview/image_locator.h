#pragma once

#include <boltview/cuda_utils.h>
#include <boltview/image_view_utils.h>

namespace bolt {

/// \addtogroup Views
/// @{
/// \addtogroup ElementAccess
/// @{

/// Possible ways to handle boundary problem
enum class BorderHandling {
	kNone, //< No handling - accessing outside element has undefined behavior
	kMirror, //< Mirror the image outside the boundary
	kRepeat, //< Repeat boundary value (zero derivation)
	kPeriodic, //< Periodic extension of the image
	kZero //< Return zero element for out of boundary access
};

/// Border handling traits
BOLT_HD_WARNING_DISABLE
template <BorderHandling tBorderHandling>
struct BorderHandlingTraits {
	static constexpr BorderHandling kValue = tBorderHandling;

	BOLT_HD_WARNING_DISABLE
	template<typename TView>
	BOLT_DECL_HYBRID
	static typename TView::Element access(
				const TView &view,
				const typename TView::IndexType &coordinates,
				const Vector<typename TView::TIndex, TView::kDimension> &offset)
	{
		return view[coordinates + offset];
	}
};

BOLT_HD_WARNING_DISABLE
template <>
struct BorderHandlingTraits<BorderHandling::kMirror> {
	static constexpr BorderHandling kValue = BorderHandling::kMirror;

	BOLT_HD_WARNING_DISABLE
	template<typename TView>
	BOLT_DECL_HYBRID
	static typename TView::Element access(
				const TView &view,
				const typename TView::IndexType &coordinates,
				const Vector<typename TView::TIndex, TView::kDimension> &offset)
	{
		using IndexType = typename TView::IndexType;
		auto region = validRegion(view);
		auto minimum = region.corner; //IndexType();
		auto maximum = minimum + region.size - IndexType::fill(1);
		auto coords_in_view = coordinates + offset;
		for (int i = 0; i < TView::kDimension; ++i) {
			if (coords_in_view[i] < minimum[i]) {
				coords_in_view[i] = minimum[i] + (minimum[i] - coords_in_view[i]);
			} else {
				if (coords_in_view[i] > maximum[i]) {
					coords_in_view[i] = maximum[i] - (coords_in_view[i] - maximum[i]);
				}
			}
		}
		return view[coords_in_view];
	}
};

BOLT_HD_WARNING_DISABLE
template <>
struct BorderHandlingTraits<BorderHandling::kRepeat> {
	static constexpr BorderHandling kValue = BorderHandling::kRepeat;

	BOLT_HD_WARNING_DISABLE
	template<typename TView>
	BOLT_DECL_HYBRID
	static typename TView::Element access(
				const TView &view,
				const typename TView::IndexType &coordinates,
				const Vector<typename TView::TIndex, TView::kDimension> &offset)
	{
		using IndexType = typename TView::IndexType;
		auto region = validRegion(view);
		auto minimum = region.corner; //IndexType();
		auto maximum = region.size - IndexType::fill(1);
		auto coords = min(maximum, max(minimum, coordinates + offset));
		return view[coords];
	}
};


BOLT_HD_WARNING_DISABLE
template<>
struct BorderHandlingTraits<BorderHandling::kZero> {
	static constexpr BorderHandling kValue = BorderHandling::kZero;

	BOLT_HD_WARNING_DISABLE
	template<typename TView>
	BOLT_DECL_HYBRID static typename TView::Element access(const TView &view,
														   const typename TView::IndexType &coordinates,
														   const Vector<typename TView::TIndex, TView::kDimension> & /*offset*/) {
		if (view.isIndexInside(coordinates)) {
			return view[coordinates];
		}
		auto region = validRegion(view);
		auto minimum = region.corner;

		return 0 * view[minimum];
	}
};


// TODO(johny) the rest of the border handling strategies

/// Locators serve as relative accessors to image view data.
/// It is constructed by specifiing image view and n-d index.
/// You can access image view elements on position relative to specified index by providing n-d offset.
/// \tparam TImageView Type of the wrapped image view
/// \tparam TBorderHandling How we handle the boundary problem (BorderHandlingTraits)
BOLT_HD_WARNING_DISABLE
template<typename TImageView, typename TBorderHandling>
class ImageLocator
{
public:
	constexpr static int kDimension = TImageView::kDimension;
	using SizeType = typename TImageView::SizeType;
	using IndexType = typename TImageView::IndexType;
	using OffsetType = Vector<typename TImageView::TIndex, TImageView::kDimension>;
	using AccessType = typename TImageView::AccessType;
	using Element = typename TImageView::Element;

	BOLT_DECL_HYBRID
	ImageLocator(const TImageView &view, IndexType coords)
		: view_(view),
		coords_(coords)
	{
		// static_assert(!std::is_reference<typename TImageView::AccessType>::value || std::is_const<typename TImageView::AccessType>::value,
		// 	"This locator should be constructed from constant view. If you try modifying the view you will get race conditions.");
	}

	/// \return Image element on position coords() + offset
	BOLT_HD_WARNING_DISABLE
	template<typename TOffsetType>
	BOLT_DECL_HYBRID Element
	operator[](TOffsetType offset) const
	{
		return TBorderHandling::access(view_, coords_, offset);
		//TODO(johny)
		//IndexType coords = Min(view_.size()-IndexType::fill(1), Max(IndexType(), coords_ + offset));
		//return view_[coords];
	}

	/// \return Element this locator is centered on.
	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID AccessType
	get() const
	{
		return view_[coords_];
	}

	BOLT_DECL_HYBRID AccessType
	get(IndexType index) const
	{
		return view_[index];
	}

	template<typename TCoordinates>
	BOLT_DECL_HYBRID Element
	accessInterpolated(TCoordinates coordinates) const
	{
		//typename TImageView::CoordinateType
		return view_.access(coordinates);
	}


	/// gets element by offsetting the index in specified dimension only. Dimension specified in compile time - compile time error checking of correct dimensionality.
	/// \param offset How many steps in specified dimension we are offsetting the index.
	/// \tparam tDimIdx Index of dimension we are offsetting
	/// \return Element on offseted coordinates
	template <int tDimIdx>
	BOLT_DECL_HYBRID Element//AccessType
	dimOffset(int offset) const
	{
		static_assert(tDimIdx >= 0 && tDimIdx < TImageView::kDimension, "Wrong dimension index - check dimensionality of the wrapped view.");
		IndexType coords;
		coords[tDimIdx] = offset;

		return TBorderHandling::access(view_, coords_, coords);
		//coords = Min(view_.size()-IndexType::fill(1), Max(IndexType(), coords));
		//return view_[coords];
	}

	/// Coordinates the locator is anchored to.
	BOLT_DECL_HYBRID IndexType
	coords() const
	{
		return coords_;
	}

	/// Size of the wrapped view
	BOLT_DECL_HYBRID SizeType
	size() const
	{
		return view_.size();
	}

protected:
	TImageView view_;
	IndexType coords_;
};

/// @}

/** \ingroup  traits
 * @{
 **/

//template<typename TImageView, typename TBorderHandling>
//struct dimension<image_locator<TImageView, TBorderHandling> >: dimension<TImageView> {};

/**
 * @}
 **/

template<BorderHandling tBorderHandling>
struct LocatorConstruction {
	template<typename TView>
	BOLT_DECL_HYBRID
        // there was a cryptic problem when called like create(image.constView(),...)
        // it was due to binding temporary object to TView&
        // binding temprorary to const TView & should be OK in C++ standard
        // hopefuly CUDA does the same thing
	static ImageLocator<TView, BorderHandlingTraits<tBorderHandling>> create(const TView &view, typename TView::IndexType index) {
		return ImageLocator<TView, BorderHandlingTraits<tBorderHandling>>(view, index);
	}
};

/// @}
/// @}

}  // namespace bolt
