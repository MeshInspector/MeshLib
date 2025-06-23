#pragma once

#include "MRCuda.cuh"

#include "MRMesh/MRVector.h"

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

    DynamicArray( DynamicArray&& other );
    DynamicArray& operator=( DynamicArray&& other );

    DynamicArray( const DynamicArray& ) = delete;
    DynamicArray& operator=( const DynamicArray& other ) = delete;

    // copy given vector to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    template <typename U>
    cudaError_t fromVector( const std::vector<U>& vec );

    // copy given vector to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    template <typename U, typename I>
    cudaError_t fromVector( const MR::Vector<U, I>& vec )
    {
        return fromVector( vec.vec_ );
    }

    // copy given data to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    cudaError_t fromBytes( const uint8_t* data, size_t numBytes );

    // copy given data to CPU (data should be already allocated)
    cudaError_t toBytes( uint8_t* data );

    // copy this GPU array to given vector
    template <typename U>
    cudaError_t toVector( std::vector<U>& vec ) const;

    // copy given data to GPU (truncated if the array size is smaller that the data one)
    template <typename U>
    cudaError_t copyFrom( const U* data, size_t size );

    // copy given data to CPU (truncated if the data size is smaller that the array one)
    template <typename U>
    cudaError_t copyTo( U* data, size_t size ) const;

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

// Sets all float values of GPU array to zero
cudaError_t setToZero( DynamicArrayF& devArray );

}

}

#include "MRCudaBasic.hpp"