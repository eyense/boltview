// Copyright 2015 Eyen SE
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se

#pragma once

#ifdef __CUDACC__
#	include <boltview/device_image_view.h>
#endif  // __CUDACC__

#include <boltview/host_image_view.h>
#include <boltview/region.h>


namespace bolt {

BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
Region<TView::kDimension, typename TView::TIndex> validRegion(const TView &view) {
	return Region<TView::kDimension, typename TView::TIndex>{ typename TView::IndexType(), view.size() };
}

BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
auto dataSize(const TView &view) {
	return view.size();
}

template<typename TView>
struct DataDimension {
	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr int value = TView::kDimension;
};

BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
bool isEmpty(const TView &view) {
	// return 0 == product(view.size());
	return 0 == view.elementCount();
}

template<typename TView>
struct IsDeviceImageView {
	using yes = char (&)[1];  // NOLINT
	using no = char (&)[2];  // NOLINT

	template <typename TT> static yes check(decltype(TT::kIsDeviceView));
	template <typename> static no check(...);

	template <bool tHasFlag, bool tDummy>
	struct GetFlagValue {
		// NOLINTNEXTLINE(readability-identifier-naming)
		static constexpr bool value = false;
	};

	template <bool tDummy>
	struct GetFlagValue<true, tDummy> {
		// NOLINTNEXTLINE(readability-identifier-naming)
		static constexpr bool value = TView::kIsDeviceView;
	};

	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = GetFlagValue<(sizeof(check<TView>(true)) == sizeof(yes)), true>::value;
};

template<typename TView>
struct IsHostImageView {
	using yes = char (&)[1];  // NOLINT
	using no = char (&)[2];  // NOLINT

	template <typename TT> static yes check(decltype(TT::kIsHostView));
	template <typename> static no check(...);

	template <bool tHasFlag, bool tDummy>
	struct GetFlagValue {
		// NOLINTNEXTLINE(readability-identifier-naming)
		static constexpr bool value = false;
	};

	template <bool tDummy>
	struct GetFlagValue<true, tDummy> {
		// NOLINTNEXTLINE(readability-identifier-naming)
		static constexpr bool value = TView::kIsHostView;
	};

	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = GetFlagValue<(sizeof(check<TView>(true)) == sizeof(yes)), true>::value;
};

template<typename TView>
struct IsImageView {
	// NOLINTNEXTLINE(readability-identifier-naming)
	static constexpr bool value = IsHostImageView<TView>::value || IsDeviceImageView<TView>::value;
};

}  // namespace bolt
