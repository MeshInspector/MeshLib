#pragma once

#include "MRRectIndexer.h"
#include <vector>

namespace MR
{

/// \defgroup MatrixGroup Matrix
/// \ingroup MathGroup

/// Row-major matrix with T values
/// \ingroup MatrixGroup
template <typename T>
struct Matrix : RectIndexer
{
public:
    using ValueType = T;
    constexpr Matrix() noexcept = default;

    Matrix( size_t numRows, size_t numCols ) : RectIndexer( { (int)numCols, (int)numRows } )
    {
        data_.resize( size_ );
    }

    /// main access method
    constexpr T& operator ()( size_t row, size_t col ) noexcept { return data_[toIndex( { (int)col, (int)row } )]; }
    constexpr T& operator ()( size_t i ) noexcept { return data_[i]; }
    constexpr const T& operator ()( size_t row, size_t col ) const noexcept { return data_[toIndex( { (int)col, (int)row } )]; }
    constexpr const T& operator ()( size_t i ) const noexcept { return data_[i]; }

    constexpr Matrix getSubMatrix( size_t startRow, size_t nRow, size_t startCol, size_t nCol )
    {
        Matrix res(nRow, nCol);
        for( size_t r = 0; r < nRow; r++ )
        {
            for( size_t c = 0; c < nCol; c++ )
            {
                res(r, c) = data_[(startRow + r) * size_t(dims_.x) + startCol + c];
            }
        }
        return res;
    }

    /// computes transposed matrix
    constexpr Matrix transposed() const
    {
        Matrix res( dims_.x, dims_.y );
        for( size_t r = 0; r < dims_.y; r++ )
        {
            for( size_t c = 0; c < dims_.x; c++ )
            {
                res(c, r) = data_[r][c];
            }
        }
        return res;
    }

    void fill( T val )
    {
        for( auto& elem : data_ )
        {
            elem = val;
        }
    }

    void clear()
    {
        RectIndexer::resize( {0, 0} );
        data_.clear();
    }

    size_t getRowsNum() const
    {
        return dims_.y;
    }

    size_t getColsNum() const
    {
        return dims_.x;
    }

    const std::vector<T> & data() const
    {
        return data_;
    }

private:
    std::vector<T> data_;
};

} // namespace MR
