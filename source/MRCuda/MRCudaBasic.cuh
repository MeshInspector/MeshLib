#pragma once

#include "cuda_runtime.h"
#include <cstdint>
#include <vector>

namespace MR
{

namespace Cuda
{
// This struct is present to simplify GPU memory control
template<typename T>
class DynamicArray
{
public:
    DynamicArray() = default;
    // malloc given size on GPU
    explicit DynamicArray( size_t size );
    // copy given vector to GPU
    template <typename U>
    explicit DynamicArray( const std::vector<U>& vec );
    // free this array from GPU (if needed)
    ~DynamicArray();

    DynamicArray( const DynamicArray& ) = delete;
    DynamicArray( DynamicArray&& ) = delete;
    DynamicArray& operator=( DynamicArray&& ) = delete;
    DynamicArray& operator=( const DynamicArray& other ) = delete;

    // copy given vector to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    template <typename U>
    cudaError_t fromVector( const std::vector<U>& vec );

    // copy given data to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    template <typename U>
    cudaError_t copyFrom( const U* data, size_t size );

    // copy given data to CPU (data should be already allocated)
    template <typename U>
    cudaError_t copyTo( U* data, size_t size ) const;

    // copy this GPU array to given vector
    template <typename U>
    cudaError_t toVector( std::vector<U>& vec ) const;

    // resize (free and malloc againg if size inconsistent) this GPU array (if size == 0 free it (if needed))
    cudaError_t resize( size_t size );

    // pointer to GPU array
    T* data()
    {
        return data_;
    }
    // const pointer to GPU array
    const T* data() const
    {
        return data_;
    }
    // size of GPU array in elements
    size_t size() const
    {
        return size_;
    }
    // size of GPU array in bytes
    size_t bytes() const
    {
        return size_ * sizeof( T );
    }

private:
    T* data_{ nullptr };
    size_t size_{ 0 };
};

using DynamicArrayU64 = MR::Cuda::DynamicArray<uint64_t>;
using DynamicArrayU16 = MR::Cuda::DynamicArray<uint16_t>;
using DynamicArrayF = MR::Cuda::DynamicArray<float>;

// ...
template <typename T>
class BufferSlice
{
public:
    BufferSlice() = default;

    static size_t maxGroupCount( size_t maxBytes, size_t groupSize );

    // resize the underlying buffer
    cudaError_t allocate( size_t size ) { return buf_.resize( size ); }

    // ...
    cudaError_t release() { return buf_.resize( 0 ); }

    // ...
    void assignOutput( T* data, size_t size ) { outData_ = data; outSize_ = size; }
    template <typename U>
    void assignOutput( std::vector<U>& vec );

    // ...
    void setOverlap( size_t overlap );

    // move the slice window
    void advance();

    // ...
    size_t offset() const { return offset_; }

    // ...
    bool valid() const { return outData_ != nullptr; }

    // ...
    cudaError_t copyToOutput() const;

    // ...
    T* data() { return buf_.data(); }
    // ...
    const T* data() const { return buf_.data(); }

    // ...
    size_t size() const { return std::min( buf_.size(), outSize_ ); }

    // ...
    size_t bytes() const { return buf_.bytes(); }

private:
    DynamicArray<T> buf_;

    T* outData_{ nullptr };
    size_t outSize_{ 0 };

    size_t overlap_{ 0 };
    size_t offset_{ 0 };
};

// Sets all float values of GPU array to zero
cudaError_t setToZero( DynamicArrayF& devArray );

}

}

#include "MRCudaBasic.hpp"