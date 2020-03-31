// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#include <type_traits>
#include <typeinfo>

#include <boltview/cuda_utils.h>
#include <boltview/host_image_view.h>
#include <boltview/region.h>

#include <boltview/exception_error_info.h>

#include <boltview/device_image_view_base.h>

#if defined(__CUDACC__)
#include <boltview/device_image_view.h>
#endif  // __CUDACC__

namespace bolt {

/// \addtogroup Views
/// @{

namespace detail {

template<typename TSizeType, typename TIndexType, bool tKSameTypes = std::is_same<TSizeType, TIndexType>::value>
struct AccessHelper {};

template<typename TSizeType, typename TIndexType>
struct AccessHelper<TSizeType, TIndexType, false> {
	using type = TSizeType;
};

template<typename TSizeType, typename TIndexType>
struct AccessHelper<TSizeType, TIndexType, true> {
	using type = int;
};

}  // namespace detail

BOLT_HD_WARNING_DISABLE
template<typename TView>
class SubImageView: public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	static const int kDimension = TView::kDimension;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;


	BOLT_DECL_HYBRID
	SubImageView(TView view, const IndexType &corner, const SizeType &size) :
		Predecessor(size),
		view_(view),
		corner_(corner)
	{
		#ifndef __CUDA_ARCH__
		BOLT_DFORMAT("Subview of %1%, %2%, %3%", typeid(view).name(), corner, size);
		#endif
	}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		index += corner_;
		return view_[index];
	}

	BOLT_DECL_HYBRID
	AccessType access(Vector<float, kDimension> coords) const {
		return view_.access(corner_ + coords);
	}

	BOLT_DECL_HYBRID
	const TView &parentView() const {
		return view_;
	}

	BOLT_DECL_HYBRID
	const IndexType &corner() const {
		return corner_;
	}

protected:
	TView view_;
	IndexType corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};


BOLT_HD_WARNING_DISABLE
template<typename TView>
class BorderedSubImageView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	static const int kDimension = TView::kDimension;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;

	BorderedSubImageView(TView view, const IndexType &corner, const SizeType &size) :
		Predecessor(size),
		view_(view),
		corner_(corner)
	{
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		index += corner_;
		return view_[index];
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType access(Vector<float, kDimension> coords) const {
		return view_.access(corner_ + coords);
	}

	BOLT_DECL_HYBRID
	const TView &parentView() const {
		return view_;
	}

	BOLT_DECL_HYBRID
	const IndexType &corner() const {
		return corner_;
	}

protected:
	TView view_;
	Vector<int, kDimension> corner_;  //< Index of the wrapper view topleft corner in the wrapped view
};

BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
Region<TView::kDimension, typename TView::TIndex> validRegion(const BorderedSubImageView<TView> &view) {
	auto region = validRegion(view.parentView());
	region.corner -= view.corner();
	return region;
}

/// Wrapper providing access to the cut through the wrapped view.
/// \tparam tSliceDimension Which axis is perpendicular to the cut.
template<typename TView, int tSliceDimension, bool tIsDeviceView>
class SliceImageView;

#if defined(__CUDACC__)
template<typename TView, int tSliceDimension>
class SliceImageView<TView, tSliceDimension, true> : public DeviceImageViewBase<TView::kDimension - 1, typename TView::Policy> {
public:
	static const bool kIsDeviceView = true;
	static const int kDimension = TView::kDimension - 1;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using Predecessor = DeviceImageViewBase<TView::kDimension - 1, Policy>;
	using SizeType = typename DecreaseDimension<typename TView::SizeType>::type;
	using IndexType = typename DecreaseDimension<typename TView::IndexType>::type;
	using IndexElement = typename IndexType::Element;
	using SizeElement = typename SizeType::Element;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;

	SliceImageView(TView view, IndexElement slice) :
		Predecessor(removeDimension(view.size(), tSliceDimension)),
		view_(view),
		slice_coordinate_(slice)
	{}

	BOLT_DECL_DEVICE
	AccessType operator[](IndexType index) const {
		auto new_index = insertDimension(index, slice_coordinate_, tSliceDimension);
		return view_[new_index];
	}

	BOLT_DECL_DEVICE
	AccessType operator[](typename  detail::AccessHelper<SizeType, IndexType>::type index) const {
		return this->operator[](static_cast<IndexType>(index));
	}

protected:
	TView view_;
	IndexElement slice_coordinate_;
};
#endif  // __CUDACC__

template<typename TView, int tSliceDimension>
class SliceImageView<TView, tSliceDimension, false> : public HostImageViewBase<TView::kDimension - 1, typename TView::Policy> {
public:
	static const bool kIsDeviceView = false;
	static const int kDimension = TView::kDimension - 1;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using Predecessor = HostImageViewBase<TView::kDimension - 1, Policy>;
	using SizeType = typename Predecessor::SizeType;
	using IndexType = typename Predecessor::IndexType;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;

	SliceImageView(TView view, int slice) :
		Predecessor(removeDimension(view.size(), tSliceDimension)),
		view_(view),
		slice_coordinate_(slice)
	{}

	AccessType operator[](IndexType index) const {
		auto new_index = insertDimension(index, slice_coordinate_, tSliceDimension);
		return view_[new_index];
	}

protected:
	TView view_;
	int slice_coordinate_;
};


namespace detail {

template<typename TView, bool tIsMemoryBased>
struct SubviewGenerator {
	using ResultView = SubImageView<TView>;

	static ResultView invoke(TView view, const typename TView::IndexType &corner, const typename TView::SizeType &size) {
		return ResultView(view, corner, size);
	}
};


template<typename TView>
struct SubviewGenerator<TView, true> {
	using ResultView = decltype(std::declval<TView>().subview(typename TView::IndexType(), typename TView::SizeType()));

	static ResultView invoke(TView view, const typename TView::IndexType &corner, const typename TView::SizeType &size) {
		return view.subview(corner, size);
	}
};


template<typename TView, int tSliceDimension, bool tIsMemoryBased>
struct SliceGenerator {
	using ResultView = SliceImageView<TView, tSliceDimension, TView::kIsDeviceView>;

	static ResultView invoke(TView view, typename TView::IndexType::Element slice) {
		return ResultView(view, slice);
	}
};


template<typename TView, int tSliceDimension>
struct SliceGenerator<TView, tSliceDimension, true> {
	using ResultView = decltype(std::declval<TView>().template slice<tSliceDimension>(0));

	static ResultView invoke(TView view, typename TView::IndexType::Element slice) {
		return view.template slice<tSliceDimension>(slice);
	}
};

}  // namespace detail

/// Creates view for part of the original image view (device or host).
/// When the original view is not memory based it returns view wrapper.
/// \param view Original image view.
/// \param corner Index of view corner (zero coordinates in new view)
/// \param size Size of the subview
template<typename TView>
auto subview(
	const TView &view,
	const typename TView::IndexType &corner,
	const typename TView::SizeType &size)
	-> typename detail::SubviewGenerator<TView, TView::kIsMemoryBased>::ResultView
{
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, view.size());
	//BOLT_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//BOLT_ASSERT(corner < view.size());
	//BOLT_ASSERT((corner + size) <= view.size());
	bool corner_inside = corner >= Vector<typename TView::TIndex, TView::kDimension>() && corner < view.size();
	if (!corner_inside || !((corner + size) <= view.size())) {
		BOLT_THROW(InvalidNDRange()
				<< getOriginalRegionErrorInfo(view.getRegion())
				<< getWrongRegionErrorInfo(createRegion(static_cast<typename TView::SizeType>(corner), size)));
	}
	return detail::SubviewGenerator<TView, TView::kIsMemoryBased/*typename Category<TView>::Type*/>::invoke(view, corner, size);
}

template<typename TView>
auto subview(
	const TView &view,
	const Region<TView::kDimension, typename TView::TIndex> &region
	)
	-> typename detail::SubviewGenerator<TView, TView::kIsMemoryBased>::ResultView
{
	return subview(view, region.corner, region.size);
}


/// Creates view for part of the original image view (device or host).
/// In behavior same as the normal subview, but read access to elements
/// outside its domain is valid, as long as it is valid in the original image view.
/// \param view Original image view.
/// \param corner Index of view corner (zero coordinates in new view)
/// \param size Size of the subview
template<typename TView>
auto borderedSubview(
	const TView &view,
	const Vector<typename TView::TIndex, TView::kDimension> &corner,
	const Vector<typename TView::TIndex, TView::kDimension> &size)
	-> BorderedSubImageView<TView> //TODO(johny) hybrid views?
{
	//D_FORMAT("Generating subview: corner: %1%, size: %2%, original size: %3%", corner, size, view.size());
	//BOLT_ASSERT((corner >= Vector<int, TView::kDimension>()));
	//BOLT_ASSERT(corner < view.size());
	//BOLT_ASSERT((corner + size) <= view.size());
	bool corner_inside = corner >= Vector<typename TView::TIndex, TView::kDimension>() && corner < view.size();
	if (!corner_inside || !((corner + size) <= view.size())) {
		BOLT_THROW(InvalidNDRange() << getOriginalRegionErrorInfo(view.getRegion()) << getWrongRegionErrorInfo(createRegion(corner, size)));
	}
	return BorderedSubImageView<TView>(view, corner, size);
}


template<typename TView>
auto borderedSubview(
	const TView &view,
	const Region<TView::kDimension, typename TView::TIndex> &region
	)
	-> BorderedSubImageView<TView>
{
	return BorderedSubview(view, region.corner, region.size);
}


/// Creates slice view (view of smaller dimension).
/// \tparam tSliceDimension Which axis is perpendicular to the cut
/// TODO(johny) - generic slicing
template<int tSliceDimension, typename TView>
auto slice(
	const TView &view,
	typename TView::IndexType::Element slice)
	-> typename detail::SliceGenerator<TView, tSliceDimension, TView::kIsMemoryBased>::ResultView
{
	BOLT_ASSERT(slice >= 0);
	BOLT_ASSERT(slice < view.size()[tSliceDimension]);
	if (slice < 0 || slice >= view.size()[tSliceDimension]) {
		BOLT_THROW(SliceOutOfRange() << getOriginalRegionErrorInfo(view.getRegion()) << WrongSliceErrorInfo(Int2(slice, tSliceDimension)));
	}
	return detail::SliceGenerator<TView, tSliceDimension, TView::kIsMemoryBased>::invoke(view, slice);
}

/// @}

}  // namespace bolt
