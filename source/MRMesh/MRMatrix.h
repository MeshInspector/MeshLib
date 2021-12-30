#pragma once
#include <vector>

namespace MR
{

// simple 2d container with linear vector implementation
template <typename T>
struct Matrix
{
public:
    using ValueType = T;
    constexpr Matrix() noexcept = default;

    Matrix( size_t numRows, size_t numCols )
    {
        data_.resize( size_t(numCols) * numRows );
        rows_ = numRows;
        cols_ = numCols;
    }

    // main access method
    constexpr T& operator ()( size_t row, size_t col ) noexcept { return data_[row * size_t(cols_) + col]; }
    constexpr T& operator ()( size_t i ) noexcept { return data_[i]; }
    constexpr const T& operator ()( size_t row, size_t col ) const noexcept { return data_[row * size_t(cols_) + col]; }
    constexpr const T& operator ()( size_t i ) const noexcept { return data_[i]; }

    constexpr Matrix getSubMatrix( size_t startRow, size_t nRow, size_t startCol, size_t nCol )
    {
        Matrix res(nRow, nCol);
        for( size_t r = 0; r < nRow; r++ )
        {
            for( size_t c = 0; c < nCol; c++ )
            {
                res(r, c) = data_[(startRow + r) * size_t(cols_) + startCol + c];
            }
        }
        return res;
    }

    // computes transposed matrix
    constexpr Matrix transposed() const noexcept
    {
        Matrix res( cols_, rows_ );
        for( size_t r = 0; r < rows_; r++ )
        {
            for( size_t c = 0; c < cols_; c++ )
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
        data_.clear();
        rows_ = 0, cols_ = 0;
    }

    size_t getRowsNum() const
    {
        return rows_;
    }

    size_t getColsNum() const
    {
        return cols_;
    }

    size_t size() const
    {
        return rows_ * cols_;
    }

private:
    size_t rows_ = 0, cols_ = 0;
    std::vector<T> data_;
};

} //namespace MR
