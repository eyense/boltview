// Copyright 2015 Eyen SE
// Authors: Jan Kolomaznik jan.kolomaznik@eyen.se, Adam Kubista adam.kubista@eyen.se

#pragma once

#include <type_traits>
#include <boltview/host_image_view_base.h>
#include <boltview/image_view_utils.h>

namespace bolt {

template<typename TView>
struct IsTextureView  : std::integral_constant<bool, false> {};

template<typename TView>
struct IsInterpolatedView : std::integral_constant<bool, false> {};

template<typename TView, typename std::enable_if<std::is_class<TView>::value && std::is_pointer<decltype(&TView::kIsHostView)>::value>::type * = nullptr>
struct IsDeviceView : std::integral_constant<bool, TView::kIsDeviceView> {};

template<typename TView, typename std::enable_if<std::is_class<TView>::value && std::is_pointer<decltype(&TView::kIsHostView)>::value>::type * = nullptr>
struct IsHostView : std::integral_constant<bool, TView::kIsHostView> {};

template<typename TView>
struct IsMemoryBasedView : std::integral_constant<bool, TView::kIsMemoryBased> {};

template<typename TView>
struct IsMemcpyAble : std::integral_constant<bool, TView::kIsMemoryBased> {};

template<typename TView>
struct IsArrayView  : std::integral_constant<bool, false> {};

template<typename TTypeA, typename TTypeB, typename std::enable_if<IsImageView<TTypeA>::value && IsImageView<TTypeB>::value>::type * = nullptr>
struct AreCompatibleViews : std::integral_constant<bool,
    ((IsDeviceView<TTypeA>::value && IsDeviceView<TTypeB>::value) || (IsHostView<TTypeA>::value && IsHostView<TTypeB>::value))> {};

template <
        bool tIsDeviceView = false,
        bool tIsHostView = false,
        bool tIsMemoryBased = false,
        bool tIsTextureView = false,
        bool tIsInterpolatedView = false>
struct ViewCategory {};

template<typename TView>
struct Category {
	using Type = ViewCategory<
		TView::kIsDeviceView,
		TView::kIsHostView,
		TView::kIsMemoryBased,
		IsTextureView<TView>::value,
		IsInterpolatedView<TView>::value>;
};

}  //namespace bolt

