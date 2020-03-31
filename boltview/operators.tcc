// Copyright 2019 Eyen SE
// Author: Adam Kubista adam.kubista@eyen.se

#pragma once

#include <boltview/view_traits.h>
#include <boltview/image_view_utils.h>

namespace bolt {

		/**
		Performs per-element  subtraction of 2 compatible views.
		This is more like sugar to subtract() function
		*/
		template<typename TView1, typename TView2, typename std::enable_if<AreCompatibleViews<TView1, TView2>::value>::type * = nullptr>
		LinearCombinationImageView<int, TView1, int, TView2>
		operator-(TView1 A, TView2 B){
			return subtract(A, B);
		}

		/**
		Performs per-element addition of 2 compatible views.
		This is more like sugar to add() function
		*/
		template<typename TView1, typename TView2, typename std::enable_if<AreCompatibleViews<TView1, TView2>::value>::type * = nullptr>
		LinearCombinationImageView<int, TView1, int, TView2>
		operator+(TView1 A, TView2 B){
			return add(A, B);
		}

		/**
		*/
		template<typename TView, typename TValue, typename std::enable_if<std::is_scalar<TValue>::value && IsImageView<TView>::value>::type * = nullptr>
		auto operator+(TView A, TValue B){
			return addValue(B, A);
		}

		template<typename TView, typename TValue, typename std::enable_if<std::is_scalar<TValue>::value && IsImageView<TView>::value>::type * = nullptr>
		auto operator-(TView A, TValue B){
			return addValue(-B, A);
		}

		/**
		Performs per-element multiplication of 2 compatible views.
		This is more like sugar to multiply() function
		*/
		template<typename TView1, typename TView2, typename std::enable_if<AreCompatibleViews<TView1, TView2>::value>::type * = nullptr>
		MultiplicationImageView<TView1, TView2>
		operator*(TView1 A, TView2 B){
			return multiply(A, B);
		}

		/**
		Performs per-element multiplication by a factor.
		This is more like sugar to multiplyByFactor() function
		*/
		template<typename TFactor, typename TView, typename std::enable_if<std::is_scalar<TFactor>::value && IsImageView<TView>::value>::type * = nullptr>
		auto operator*(TFactor f, TView v){
			return multiplyByFactor(f, v);
		}

		/**
		Performs per-element multiplication by a factor.
		This is more like sugar to multiplyByFactor() function
		*/
		template<typename TFactor, typename TView, typename std::enable_if<std::is_scalar<TFactor>::value && IsImageView<TView>::value>::type * = nullptr>
		auto operator*(TView v, TFactor f){
			return multiplyByFactor(f, v);
		}


		/**
		Performs per-element division of 2 compatible views.
		This is more like sugar to Divide() function
		*/
		template<typename TView1, typename TView2, typename std::enable_if<AreCompatibleViews<TView1, TView2>::value>::type * = nullptr>
		DivisionImageView<TView1, TView2>
		operator/(TView1 A, TView2 B){
			return divide(A, B);
		}

}//namespace bolt

