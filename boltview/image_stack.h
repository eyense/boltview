#pragma once

#include <boltview/math/vector.h>
#include <boltview/detail/algorithm_common.h>
#include <boltview/detail/meta_algorithm_utils.h>

#include <boltview/subview.h>

namespace bolt {

template<int tDimension>
struct StackSize {
	Vector<int, tDimension> slice_size;
	int count;
};

template<int tDimension>
struct StackIndex {
	Vector<int, tDimension> slice_index;
	int slice;
};
using StackSize2 = StackSize<2>;
using StackSize3 = StackSize<3>;

DEFINE_VIEW_PAIR_SIZES_EXN(StackSize2);
DEFINE_VIEW_PAIR_SIZES_EXN(StackSize3);

template<int tDimension>
std::ostream &operator<<(std::ostream &s, const StackSize<tDimension> &size) {
	return s << "{ slice_size: " << size.slice_size << "; count: " << size.count << "; }";
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator!=(const StackIndex<tDimension> &a, const StackIndex<tDimension> &b) {
	return !(a == b);
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator!=(const StackSize<tDimension> &a, const StackSize<tDimension> &b) {
	return !(a == b);
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator==(const StackIndex<tDimension> &a, const StackIndex<tDimension> &b) {
	return a.slice_index == b.slice_index && a.slice == b.slice;
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator==(const StackSize<tDimension> &a, const StackSize<tDimension> &b) {
	return a.slice_size == b.slice_size && a.count == b.count;
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const StackIndex<tDimension> &a, const StackIndex<tDimension> &b) {
	return a.slice_size >= b.slice_size && a.count >= b.count;
}

template<int tDimension>
BOLT_DECL_HYBRID
bool operator>=(const StackSize<tDimension> &a, const StackSize<tDimension> &b) {
	return a.slice_size >= b.slice_size && a.count >= b.count;
}

template<int tDimension>
BOLT_DECL_HYBRID
StackSize<tDimension-1> StackSizeFromVector(const Vector<int, tDimension> &size) {
	return { removeDimension(size, tDimension-1), size[tDimension-1] };
}

template<int tDimension>
BOLT_DECL_HYBRID
StackIndex<tDimension> StackIndexFromVector(const Vector<int, tDimension + 1> &idx) {
	return { removeDimension(idx, tDimension), idx[tDimension] };
}

template<int tDimension>
BOLT_DECL_HYBRID
Vector<int, tDimension + 1> ToVector(const StackSize<tDimension> &size) {
	return InsertDimension(size.slice_size, size.count, tDimension);
}

template<int tDimension>
BOLT_DECL_HYBRID
Vector<int, tDimension + 1> ToVector(const StackIndex<tDimension> &idx) {
	return InsertDimension(idx.slice_index, idx.slice, tDimension);
}

template<typename TImageView>
class ImageStackAdapterView: protected TImageView {
public:
	static constexpr bool kIsHostView = TImageView::kIsHostView;
	static constexpr bool kIsDeviceView = TImageView::kIsDeviceView;
	static constexpr bool kIsMemoryBased = TImageView::kIsMemoryBased;
	static constexpr int kDimension = TImageView::kDimension - 1;

	using SizeType = StackSize<kDimension>;
	using IndexType = StackIndex<kDimension>;
	using Element = typename TImageView::Element;
	using AccessType = typename TImageView::AccessType;
	using Predecessor = TImageView;

	using SliceView = decltype(slice<kDimension>(std::declval<Predecessor>(), 0));

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	ImageStackAdapterView(TImageView view) :
		Predecessor(std::move(view))
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	SizeType size() const {
		return StackSizeFromVector(Predecessor::Size());
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	int64_t elementCount() const {
		return Predecessor::elementCount();
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	const Predecessor &data() const {
		return *this;
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Predecessor &data() {
		return *this;
	}

	using Predecessor::operator[];

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType idx) const {
		return Predecessor::operator[](ToVector(idx));
	}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	SliceView slice(int idx) const {
		return bolt::slice<kDimension>(data(), idx);
	}
protected:

};


/// Adapter class for the data owning images.
template<typename TImage>
class ImageStackAdapter: protected TImage {
public:
	using SizeType = typename TImage::SizeType;
	using IndexType = typename TImage::IndexType;
	using Element = typename TImage::Element;
	using Predecessor = TImage;

	using ViewType = ImageStackAdapterView<typename Predecessor::ViewType>;
	using ConstViewType = ImageStackAdapterView<typename Predecessor::ConstViewType>;

	static constexpr int kDimension = TImage::kDimension - 1;

	ImageStackAdapter() = default;

	ImageStackAdapter(SizeType size) :
		Predecessor(size)
	{}

	ImageStackAdapter(TImage &&image) :
		Predecessor(std::move(image))
	{}

	ViewType view() {
		return ViewType(Predecessor::View());
	}

	ConstViewType constView() const {
		return ConstViewType(Predecessor::ConstView());
	}

protected:

};

// template<typename TImageView1, typename TImageView2>
// void copy(ImageStackAdapterView<TImageView1> from_view, ImageStackAdapterView<TImageView2> to_view) {
// 	copy(from.data(), to.data());
// }

template<typename TImageView>
struct IsImageView<ImageStackAdapterView<TImageView>> : std::integral_constant<bool, false> {};

template <typename TFromView, typename TToView, typename TTag>
inline void AsyncCopyHelper(
	ImageStackAdapterView<TFromView> from_view,
	ImageStackAdapterView<TToView> to_view,
	TTag tag,
	cudaStream_t stream)
{
	AsyncCopyHelper(from_view.data(), to_view.data(), tag, stream);
}
// template<typename TImageView1, typename TImageView2>
// void CopyAsync(
// 	ImageStackAdapterView<TImageView1> from_view,
// 	ImageStackAdapterView<TImageView2> to_view,
// 	cudaStream_t cuda_stream)
// {
// 	CopyAsync(from_view.data(), to_view.data(), cuda_stream);
// }

BOLT_HD_WARNING_DISABLE
template<typename TImageView>
BOLT_DECL_HYBRID auto getIndexFromLinearAccessIndex(const ImageStackAdapterView<TImageView> &view, int64_t index) -> typename ImageStackAdapterView<TImageView>::IndexType {
	auto idx = getIndexFromLinearAccessIndex(view.data(), index);
	return { removeDimension(idx, idx.kDimension - 1), idx[idx.kDimension - 1] };
}

BOLT_HD_WARNING_DISABLE
template<typename TView>
BOLT_DECL_HYBRID
auto dataSize(const ImageStackAdapterView<TView> &view) {
	return view.data().size();
}

template<typename TView>
struct DataDimension<ImageStackAdapterView<TView>> {
	static constexpr int value = TView::kDimension;
};

namespace detail {

template<int tDimension>
BOLT_DECL_HYBRID
inline dim3
DefaultGridSizeForBlockDim(StackSize<tDimension> idx, dim3 block_size)
{
	return DefaultGridSizeForBlockDim(InsertDimension(idx.slice_size, idx.count, tDimension-1), block_size);
}

template<typename TView>
struct ViewIndexingLocator<ImageStackAdapterView<TView>>
{
	using View = ImageStackAdapterView<TView>;
	using SliceView = typename View::SliceView;

	SliceView view_;
	typename SliceView::IndexType location_;
public:
	BOLT_HD_WARNING_DISABLE
	ViewIndexingLocator(const SliceView & view, const typename SliceView::IndexType & location):
		view_(view),
		location_(location){};
	using AccessType = decltype(view_[typename SliceView::IndexType{}]);

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID AccessType get() const
	{
		return view_[location_];
	}

	BOLT_HD_WARNING_DISABLE
	template<typename TPolicy>
	static BOLT_DECL_HYBRID ViewIndexingLocator create(const View & view, const typename View::IndexType &location, const TPolicy & policy)
	{
		return ViewIndexingLocator{ view.slice(location.slice), location.slice_index };
	}

	BOLT_HD_WARNING_DISABLE
	template<typename TPolicy>
	static BOLT_DECL_HYBRID ViewIndexingLocator create(const View & view, const Vector<int, 3> &location, const TPolicy & policy)
	{
		return ViewIndexingLocator{ view.slice(location[View::kDimension]), removeDimension(location, View::kDimension) };
	}
};


template <typename TView, bool tSharedMemoryPreload>
struct LocatorConstructor<ImageStackAdapterView<TView>, tSharedMemoryPreload> {
	using View = ImageStackAdapterView<TView>;
	using SliceView = typename View::SliceView;

	template<typename TPolicy>
	static BOLT_DECL_HYBRID
	auto create(const View & view, const typename View::IndexType & location, const TPolicy & policy)
	{
		return LocatorConstructorImageViewImpl<SliceView, tSharedMemoryPreload>::create(view.slice(location.slice), location.slice_index, policy);
	}

	template<typename TPolicy>
	static BOLT_DECL_HYBRID
	auto create(const View & view, const Vector<int, 3>/*typename View::SliceView::IndexType*/ & location, const TPolicy & policy)
	{
		return LocatorConstructorImageViewImpl<SliceView, tSharedMemoryPreload>::create(view.slice(location[View::kDimension]), removeDimension(location, View::kDimension), policy);
	}
};

}  // namespace detail

}  // namespace bolt

