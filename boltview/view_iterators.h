// Copyright 2018 Eyen SE
// Author: Pavel Mikus pavel.mikus@eyen.se

#pragma once

#include <boltview/math/vector.h>

namespace bolt {


template <typename TType, int tDimension>
Vector<TType, tDimension> wrapToIndex(int linear_index, Vector<TType, tDimension> original_size) {
	Vector<TType, tDimension> index;
	int value = linear_index;
	auto size = original_size;
	for (int i = 0; i < tDimension; i++) {
		size[i] = 1;
		index[i] = value / product(size);
		value = value % product(size);
	}
	return index;
}

//TODO(johny) - add support for device/unified view
/// bidirectional iterator for the host views
template <typename TView>
struct ViewIterator {
	static_assert(TView::kIsHostView, "Iterators work on host views only.");
	using Element = typename TView::Element;  // value_type;
	using TIndex = typename TView::Policy::IndexType;
	using difference_type = TIndex;
	using AccessType = typename TView::AccessType;  //&reference;
	using IndexType = typename TView::IndexType;
	using iterator_category = std::random_access_iterator_tag;

	explicit ViewIterator(TView view) : view_(view), current_(product(view.size())) {}

	ViewIterator(TView view, TIndex start) : view_(view), current_(start) {}

	AccessType operator*() const { return view_[wrapToIndex(current_, view_.size())]; }

	AccessType operator[](TIndex k) const { return view_[wrapToIndex(current_ + k, view_.size())]; }

	ViewIterator& operator++() {
		current_++;
		return *this;
	}

	ViewIterator operator++(int) {
		auto tmp = *this;
		current_++;
		return tmp;
	}

	ViewIterator& operator--() {
		current_--;
		return *this;
	}

	ViewIterator operator--(int) {
		auto tmp = *this;
		current_--;
		return tmp;
	}

	ViewIterator& operator+=(TIndex k) {
		current_ += k;
		return *this;
	}

	ViewIterator& operator-=(TIndex k) {
		current_ -= k;
		return *this;
	}

	ViewIterator operator+(TIndex k) const { return ViewIterator(view_, current_ + k); }

	ViewIterator operator-(TIndex k) const { return ViewIterator(view_, current_ - k); }

	int operator-(const ViewIterator<TView>& rhs) const { return current_ - rhs.current_; }

	bool operator==(const ViewIterator<TView>& rhs) const { return (current_ == rhs.current_); }

	bool operator!=(const ViewIterator<TView>& rhs) const { return !(*this == rhs); }

	bool operator<(const ViewIterator<TView>& rhs) const { return current_ < rhs.current_; }

	bool operator>(const ViewIterator<TView>& rhs) const { return current_ > rhs.current_; }

	bool operator<=(const ViewIterator<TView>& rhs) const { return current_ <= rhs.current_; }

	bool operator>=(const ViewIterator<TView>& rhs) const { return current_ >= rhs.current_; }

	TView view_;
	TIndex current_;
};


template <typename TView>
ViewIterator<TView> begin(TView& view) {
	return ViewIterator<TView>(view, 0);
}


template <typename TView>
ViewIterator<TView> end(TView& view) {
	return ViewIterator<TView>(view, product(view.size()));
}


}  // namespace bolt
