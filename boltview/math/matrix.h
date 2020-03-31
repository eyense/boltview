#pragma once

#include <boltview/math/vector.h>

namespace bolt {

template<typename TMatrix>
class ColumnAccessor
{
public:
	BOLT_DECL_HYBRID
	auto get(int aIdx) const -> decltype(std::declval<TMatrix>().get(aIdx, 0))
	{
		return mMatrix.get(aIdx, mColumn);
	}

	BOLT_DECL_HYBRID
	auto operator[](int aIdx) const  -> decltype(std::declval<TMatrix>().get(aIdx, 0))
	{
		return mMatrix.get(aIdx, mColumn);
	}

	template<typename TVector>
	BOLT_DECL_HYBRID typename std::enable_if<static_vector_traits<TVector>::is_vector, column_accessor<TMatrix> &>::type
	operator=(const TVector &aVector)
	{
		static_assert(static_vector_traits<TVector>::dimension == TMatrix::cRowCount, "Vectors must have same dimension to be assignable");
		for (int i = 0; i < TMatrix::cRowCount; ++i) {
			mMatrix.get(i, mColumn) = aVector[i];
		}
		return *this;
	}

	int mColumn;
	TMatrix &mMatrix;
};


template <typename TType, int tRowCount, int tColCount>
class Matrix: public Vector<Vector<TType, tColCount>, tRowCount>
{
public:
	typedef Vector<Vector<TType, tColCount>, tRowCount> Base;
	static constexpr int kRowCount = tRowCount;
	static constexpr int kColCount = tColCount;

	using Element = TType;

	BOLT_DECL_HYBRID
	Matrix()
		: Base()
	{}

	BOLT_DECL_HYBRID
	TType &	get(int aRow, int aCol) {
		return (*this)[aRow][aCol];
	}

	BOLT_DECL_HYBRID
	const TType &get(int aRow, int aCol) const {
		return (*this)[aRow][aCol];
	}

	//TODO - get wrapper which provides access
	BOLT_DECL_HYBRID
	Vector<TType, tRowCount> column(int aCol) const	{
		Vector<TType, tRowCount> result;
		for (int i = 0; i < tRowCount; ++i) {
			result[i] = get(i, aCol);
		}
		return result;
	}

	BOLT_DECL_HYBRID
	ColumnAccessor<Matrix<TType, tRowCount, tColCount>>
	column(int aCol) {
		return ColumnAccessor<matrix<TType, tRowCount, tColCount>>{aCol, *this};
	}

	BOLT_DECL_HYBRID
	Vector<TType, tColCount> row(int aRow) const {
		return (*this)[aRow];
	}

	BOLT_DECL_HYBRID
	Vector<TType, tColCount> &row(int aRow)	{
		return (*this)[aRow];
	}

private:

};

template <typename TMatrix>
struct MatrixTraits
{
	static constexpr bool kIsMatrix = false;
};

template <typename TType, int tRowCount, int tColCount>
struct MatrixTraits<Matrix<TType, tRowCount, tColCount>>
{
	static constexpr bool kIsMatrix = true;
	static constexpr int kRowCount = tRowCount;
	static constexpr int kColCount = tRowCount;

	using Element = TType;
};

template <typename TType, int tRowCount>
BOLT_DECL_HYBRID TType
trace(const Matrix<TType, tRowCount, tRowCount> &aMatrix) {
	TType result = 0;
	for (int i = 0; i < tRowCount; ++i) {
		result += aMatrix.get(i, i);
	}
	return result;
}


template<typename TMatrix1, typename TMatrix2>
BOLT_DECL_HYBRID typename std::enable_if<
		Matrix_traits<TMatrix1>::kIsMatrix && MatrixTraits<TMatrix2>::kIsMatrix,
		Matrix<
			decltype(std::declval<typename MatrixTraits<TMatrix1>::Element>() * std::declval<typename MatrixTraits<TMatrix1>::Element>()),
			MatrixTraits<TMatrix1>::kRowCount,
			MatrixTraits<TMatrix2>::kColCount>
		>::type
product(const TMatrix1 &aMatrix1, const TMatrix1 &aMatrix2) {
	static_assert(MatrixTraits<TMatrix1>::kColCount == MatrixTraits<TMatrix2>::kRowCount, "Matrices do not have compatible sizes");

	Matrix<
		decltype(std::declval<typename MatrixTraits<TMatrix1>::Element>() * std::declval<typename MatrixTraits<TMatrix1>::Element>()),
		MatrixTraits<TMatrix1>::kRowCount,
		MatrixTraits<TMatrix2>::kColCount> result;

	for (int k = 0; k < MatrixTraits<TMatrix2>::kColCount; ++k) {
		for (int j = 0; j < MatrixTraits<TMatrix1>::kRowCount; ++j) {
			for (int i = 0; i < MatrixTraits<TMatrix1>::kColCount; ++i) {
				result.get(j, k) += aMatrix1.get(j, i) * aMatrix2.get(i, k);
			}
		}
	}
	return result;
}

template<typename TMatrix, typename TVector>
BOLT_DECL_HYBRID typename std::enable_if<
		MatrixTraits<TMatrix>::kIsMatrix && VectorTraits<TVector>::kIsVector,
		Vector<
			decltype(std::declval<typename MatrixTraits<TMatrix>::Element>() * std::declval<typename VectorTraits<TVector>::Element>()),
			MatrixTraits<TMatrix>::kRowCount>
		>::type
product(const TMatrix &aMatrix, const TVector &aVector) {
	static_assert(MatrixTraits<TMatrix>::kColCount == static_vector_traits<TVector>::dimension, "Matrix and vector do not have compatible sizes");

	Vector<
		decltype(std::declval<typename MatrixTraits<TMatrix>::Element>() * std::declval<typename static_vector_traits<TVector>::Element>()),
		MatrixTraits<TMatrix>::kRowCount> result;

		for (int j = 0; j < MatrixTraits<TMatrix>::kRowCount; ++j) {
			for (int i = 0; i < static_vector_traits<TVector>::dimension; ++i) {
				result[j] += aMatrix.get(j, i) * aVector[i];
			}
		}
	return result;
}

}  // namespace bolt

