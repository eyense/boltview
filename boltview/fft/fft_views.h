// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se
// Author: Jan Kolomaznik jan.kolomaznik@eyen.se
// Author: Martin Hora martin.hora@eyen.se

#pragma once

#include <boltview/procedural_views.h>

namespace bolt {

/// \addtogroup FFT
/// @{


/// \brief A view which returns result of FFT in frequency domain, centered as physicist would expect. X axis remains halved.
template<typename TView>
class HalfSpectrumView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using Policy = typename TView::Policy;
	using TIndex = typename TView::TIndex;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension, Policy>;
	using Element = typename TView::Element;
	using AccessType = typename TView::AccessType;

	explicit HalfSpectrumView(TView view) :
		Predecessor(view.size()),
		view_(view)
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	AccessType operator[](IndexType index) const {
		for (int d = 0; d < kDimension; ++d) {
			index[d] += (index[d] < 0) * get(view_.size(), d);
		}

		return view_[index];
	}

protected:
	TView view_;
};

/// Creates a view which returns result of FFT in frequency domain, centered as physicist would expect. X axis remains halved.
template<typename TView>
HalfSpectrumView<TView> halfSpectrumView(TView view) {
	return HalfSpectrumView<TView>(view);
}

/// \brief a view which returns result of FFT in frequency domain, centered, flipped and mirrored as physicist would expect.
/// The size of the ConstSpectrumView is allways a vector of odd numbers.
/// It allows accessing the assymetric positions of even-sized spectra using both positive and negative indices.
template<typename TView>
class ConstSpectrumView : public HybridImageViewBase<TView::kDimension, typename TView::Policy> {
public:
	static const int kDimension = TView::kDimension;
	static const bool kIsDeviceView = TView::kIsDeviceView;
	static const bool kIsHostView = TView::kIsHostView;
	using Policy = typename TView::Policy;
	using TIndex = typename TView::TIndex;
	using SizeType = typename TView::SizeType;
	using IndexType = typename TView::IndexType;
	using Predecessor = HybridImageViewBase<TView::kDimension>;
	using Element = typename TView::Element;
	using AccessType = typename TView::Element;

	explicit ConstSpectrumView(TView view) :
		Predecessor(fullSpectrumSize(view.size())),
		halfspectrum_view_(halfSpectrumView(view))
	{}

	BOLT_HD_WARNING_DISABLE
	BOLT_DECL_HYBRID
	Element operator[](IndexType index) const {
		// If index[0] < 0, return the complex conjugate of the halfspectrum element
		// on the centrally symmetrical position.
		int is_negative = (index[0] < 0);
		index = (1 - 2 * is_negative) * index;
		Element value = halfspectrum_view_[index];
		return value + is_negative * (conjugate(value) - value);
		
		// The implementation is slightly faster than the following equivalent:
		// if (index[0] >= 0) {
		// 	return halfspectrum_view_[index];
		// } else {
		// 	return conjugate(halfspectrum_view_[-index]);
		// }
	}

protected:
	HalfSpectrumView<TView> halfspectrum_view_;

	/// \return the size of the full fft spectrum.
	/// The size is allways a vector of odd numbers.
	static SizeType fullSpectrumSize(SizeType half_spectrum_size) {
		SizeType spectrum_size;

		if (kDimension > 0) {
			// Inflate the halved dimension.
			spectrum_size[0] = half_spectrum_size[0] * 2 - 1;
		}
		// Make all the dimensions odd.
		for (int d = 1; d < kDimension; ++d) {
			spectrum_size[d] = half_spectrum_size[d] + ((half_spectrum_size[d] + 1) & 1);
		}

		return spectrum_size;
	}
};

//// Creates a view which returns result of FFT in frequency domain, centered, flipped and mirrored as physicist would expect.
template<typename TView>
ConstSpectrumView<TView> constSpectrumView(TView view) {
	return ConstSpectrumView<TView>(view);
}

template<typename TView>
BOLT_DECL_HYBRID typename HalfSpectrumView<TView>::IndexType
topCorner(const HalfSpectrumView<TView> & view) {
	using IndexType = typename HalfSpectrumView<TView>::IndexType;
	auto result = IndexType();
	for(int i = 1; i < HalfSpectrumView<TView>::kDimension; i++){
		result[i] = -(get(view.size(), i) - 1)/ 2;
	}
	return result;
}

template<typename TView>
BOLT_DECL_HYBRID typename ConstSpectrumView<TView>::IndexType
topCorner(const ConstSpectrumView<TView> & view) {
	using IndexType = typename ConstSpectrumView<TView>::IndexType;
	auto result = IndexType();
	for(int i = 0; i < ConstSpectrumView<TView>::kDimension; i++){
		result[i] = -get(view.size(), i)/2;
	}
	return result;
}

/// Border handling traits for ConstSpectrumView.
BOLT_HD_WARNING_DISABLE
struct ConstSpectrumBorderHandling {

	BOLT_HD_WARNING_DISABLE
	template<typename TView>
	BOLT_DECL_HYBRID
	static typename TView::Element access(
		const ConstSpectrumView<TView>& const_spectrum,
		const typename TView::IndexType& coordinates,
		const Vector<typename TView::TIndex, TView::kDimension>& offset
	) {
		auto coords = coordinates + offset;

		// Check that 'coords' are inside the const spectrum.
		auto corner = topCorner(const_spectrum);
		int inside = 1;
		for (int d = 0; d < TView::kDimension; ++d) {
			inside &= (coords[d] >= corner[d]);
			inside &= (coords[d] <= -corner[d]);
		}

		// Does not work for the empty spectrum, but it should not be an issue.
		return inside * const_spectrum[inside * coords];
	}
};

/// @}

template<typename TView>
struct IsMemcpyAble<HalfSpectrumView<TView>> : std::integral_constant<bool, false> {};

template<typename TView>
struct IsMemcpyAble<ConstSpectrumView<TView>> : std::integral_constant<bool, false> {};

/// \addtogroup FFT
/// @{


/// Executes per-element phase shift, expects View elements to be of cufftComplex or HostComplexType
template<typename TView>
UnaryOperatorWithMetadataImageView<TView, PhaseShiftFunctor<TView::kDimension>>
phaseShift(TView frequency_domain_view, typename TView::SizeType space_domain_size, Vector<float, TView::kDimension> coordinate_shift) {
	return UnaryOperatorWithMetadataImageView<TView, PhaseShiftFunctor<TView::kDimension>>(frequency_domain_view, PhaseShiftFunctor<TView::kDimension>(coordinate_shift, space_domain_size));
}

/// Executes per-element phase shift, expects View elements to be of cufftComplex or HostComplexType
template<typename TView, typename std::enable_if<TView::kDimension == 1>::type * = nullptr>
UnaryOperatorWithMetadataImageView<TView, PhaseShiftFunctor<TView::kDimension>>
phaseShift(TView frequency_domain_view, int space_domain_size, Vector<float, TView::kDimension> coordinate_shift) {
	return UnaryOperatorWithMetadataImageView<TView, PhaseShiftFunctor<TView::kDimension>>(frequency_domain_view, PhaseShiftFunctor<TView::kDimension>(coordinate_shift, space_domain_size));
}

/// Functor returning amplitude of complex numbers
struct FFTAmplitudeFunctor {
	BOLT_DECL_HYBRID
	float operator()(cufftComplex in_value) const {
		return (in_value.x * in_value.x) + (in_value.y * in_value.y);
	}
	float operator()(HostComplexType in_value) const {
		return (in_value.x * in_value.x) + (in_value.y * in_value.y);
	}
};

/// Generate view returning only amplitude values of complex numbers
template<typename TView>
UnaryOperatorImageView<TView, FFTAmplitudeFunctor>
amplitude(TView view) {
	return UnaryOperatorImageView<TView, FFTAmplitudeFunctor>(view, FFTAmplitudeFunctor());
}

/// Functor returning log(1+x) transformation of the passed value.
struct PerceptualScaleFunctor {
	BOLT_DECL_HYBRID
	float operator()(float in_value) const {
		return log1pf(in_value);
	}
};

/// log(1+x) transformation of the values - usefull for fft amplitude visualization
template<typename TView>
UnaryOperatorImageView<TView, PerceptualScaleFunctor>
perceptualScale(TView view) {
	return UnaryOperatorImageView<TView, PerceptualScaleFunctor>(view, PerceptualScaleFunctor());
}

/// @}
}  // namespace bolt
