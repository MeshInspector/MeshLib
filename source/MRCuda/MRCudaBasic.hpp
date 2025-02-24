#pragma once

#include "MRCudaBasic.cuh"
#include "exports.h"
#include <cassert>
#include <string>

namespace MR
{

namespace Cuda
{

/// converts given code in user-readable error string
[[nodiscard]] MRCUDA_API std::string getError( cudaError_t code );

/// spdlog::error the information about some CUDA error including optional filename and line number
MRCUDA_API cudaError_t logError( cudaError_t code, const char * file = nullptr, int line = 0 );

/// evaluates given expression and logs the error if any
#define CUDA_LOGE( expr ) MR::Cuda::logError( expr, __FILE__ , __LINE__ )
// deprecated
#define CUDA_EXEC( expr ) CUDA_LOGE( expr )

/// evaluates given expression, logs if it fails and returns error code
#define CUDA_LOGE_RETURN( expr ) if ( auto code = CUDA_LOGE( expr ); code != cudaError::cudaSuccess ) return code
// deprecated
#define CUDA_EXEC_RETURN( expr ) CUDA_LOGE_RETURN( expr )

/// if given expression evaluates to not cudaError::cudaSuccess, then returns MR::unexpected with the error string without logging
#define CUDA_RETURN_UNEXPECTED( expr ) if ( auto code = ( expr ); code != cudaError::cudaSuccess ) return MR::unexpected( MR::Cuda::getError( code ) )

/// evaluates given expression, logs if it fails and returns MR::unexpected with the error string
#define CUDA_LOGE_RETURN_UNEXPECTED( expr ) if ( auto code = CUDA_LOGE( expr ); code != cudaError::cudaSuccess ) return MR::unexpected( MR::Cuda::getError( code ) )

template<typename T>
DynamicArray<T>::DynamicArray( size_t size )
{
    resize( size );
}

template<typename T>
template<typename U>
DynamicArray<T>::DynamicArray( const std::vector<U>& vec )
{
    fromVector( vec );
}

template<typename T>
DynamicArray<T>::~DynamicArray()
{
    resize( 0 );
}

template<typename T>
template<typename U>
inline cudaError_t DynamicArray<T>::fromVector( const std::vector<U>& vec )
{
    static_assert ( sizeof( T ) == sizeof( U ) ); 
    if ( auto code = resize( vec.size() ) )
        return code;
    return CUDA_LOGE( cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice ) );
}


template <typename T>
inline cudaError_t DynamicArray<T>::fromBytes( const uint8_t* data, size_t numBytes )
{
    assert( numBytes % sizeof( T ) == 0 );
    resize( numBytes / sizeof( T ) );
    return CUDA_LOGE( cudaMemcpy( data_, data, numBytes, cudaMemcpyHostToDevice ) );
}

template <typename T>
inline cudaError_t DynamicArray<T>::toBytes( uint8_t* data, size_t numBytes ) const
{
    assert( numBytes % sizeof( T ) == 0 );
    return CUDA_LOGE( cudaMemcpy( data, data_, std::min( size_ * sizeof( T ), numBytes ), cudaMemcpyDeviceToHost ) );
}

template<typename T>
cudaError_t DynamicArray<T>::resize( size_t size )
{
    if ( size == size_ )
        return cudaSuccess;
    if ( size_ != 0 )
    {
        if ( auto code = CUDA_LOGE( cudaFree( data_ ) ) )
            return code;
    }

    size_ = size;
    if ( size_ != 0 )
    {
        if ( auto code = CUDA_LOGE( cudaMalloc( ( void** )&data_, size_ * sizeof( T ) ) ) )
            return code;
    }
    return cudaSuccess;
}

template <typename T>
size_t BufferSlice<T>::maxGroupCount( size_t maxBytes, size_t groupSize )
{
    const size_t maxElementCount = maxBytes / sizeof( T );
    return maxElementCount / groupSize;
}

template <typename T>
template <typename U>
void BufferSlice<T>::assignOutput( std::vector<U>& vec )
{
    static_assert( sizeof( U ) == sizeof( T ) );
    outData_ = vec.data();
    outSize_ = vec.size();
}

template <typename T>
void BufferSlice<T>::setOverlap( size_t overlap )
{
    assert( overlap < buf_.size() );
    overlap_ = overlap;
}

template <typename T>
void BufferSlice<T>::advance()
{
    const size_t shift = buf_.size() - overlap_;
    if ( shift < outSize_ )
    {
        offset_ += shift;
        outData_ += shift;
        outSize_ -= shift;
    }
    else
    {
        offset_ += outSize_;
        outData_ = nullptr;
        outSize_ = 0;
    }
}

template <typename T>
cudaError_t BufferSlice<T>::copyToOutput() const
{
    return buf_.toBytes( ( uint8_t* )outData_, outSize_ * sizeof( T ) );
}

template<typename T>
template<typename U>
cudaError_t DynamicArray<T>::toVector( std::vector<U>& vec ) const
{
    static_assert ( sizeof( T ) == sizeof( U ) );
    vec.resize( size_ );
    return CUDA_LOGE( cudaMemcpy( vec.data(), data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost ) );
}

inline cudaError_t setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return cudaSuccess;
    return CUDA_LOGE( cudaMemset( devArray.data(), 0, devArray.size() * sizeof( float ) ) );
}

} // namespace Cuda

} // namespace MR
