#pragma once

#include <type_traits>
#include <typeinfo>

#include <boltview/cuda_utils.h>
#include <boltview/host_image_view.h>
#include <boltview/interpolation.h>
#include <boltview/region.h>
#include <boltview/view_traits.h>

#include <boltview/device_image_view_base.h>
#if defined(__CUDACC__)
#include <boltview/device_image_view.h>
#endif  // __CUDACC__

namespace bolt {

/// View which interpolates elements of another image view, using specified interpolator
/// The interpolator must contain static function
/// TView::Element compute(const TView &view, const Vector<float, TView::kDimension> &position)
template<typename TView, typename TInterpolator>
class InterpolatedView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
	public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using Policy = typename TView::Policy;
	using TIndex = typename Policy::IndexType;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Element = typename TView::Element;
	using AccessType = typename TView::Element;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	explicit InterpolatedView(TView view) : view_(view), Predecessor(view.size()) {}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element access(Vector<float, kDimension> coords) const {
		return TInterpolator::compute(view_, coords);
	}

	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		return TInterpolator::boundedAccess(view_, index);
	}

	protected:
	TView view_;
};

BOLT_HD_WARNING_DISABLE
template<typename TInterpolator, typename TView>
BOLT_DECL_HYBRID InterpolatedView<TView, TInterpolator> makeInterpolatedView(TView view) {
	return InterpolatedView<TView, TInterpolator>(view);
}

BOLT_HD_WARNING_DISABLE
template<typename TInterpolator, typename TView>
BOLT_DECL_HYBRID InterpolatedView<TView, TInterpolator> makeInterpolatedView(TView view, TInterpolator /*interpolator*/) {
	return InterpolatedView<TView, TInterpolator>(view);
}

template<typename TView, typename TInterpolator>
struct IsInterpolatedView<InterpolatedView<TView, TInterpolator>> : std::integral_constant<bool, true> {};

}  // namespace bolt
