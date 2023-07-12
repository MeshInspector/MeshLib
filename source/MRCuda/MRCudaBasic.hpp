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

/// executes given CUDA function and checks the error code after
#define CUDA_EXEC( func )\
{\
    auto code = func; \
    if( code != cudaSuccess ) \
        logError( code, __FILE__ , __LINE__ );\
}

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
    return logError( cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice ), __FILE__ , __LINE__ );
}


template <typename T>
inline cudaError_t DynamicArray<T>::fromBytes( const uint8_t* data, size_t numBytes )
{
    assert( numBytes % sizeof( T ) == 0 );
    resize( numBytes / sizeof( T ) );
    return logError( cudaMemcpy( data_, data, numBytes, cudaMemcpyHostToDevice ), __FILE__ , __LINE__ );
}

template <typename T>
inline cudaError_t DynamicArray<T>::toBytes( uint8_t* data )
{
    return logError( cudaMemcpy( data, data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost ), __FILE__ , __LINE__ );
}

template<typename T>
cudaError_t DynamicArray<T>::resize( size_t size )
{
    if ( size == size_ )
        return cudaSuccess;
    if ( size_ != 0 )
    {
        if ( auto code = logError( cudaFree( data_ ), __FILE__ , __LINE__ ) )
            return code;
    }

    size_ = size;
    if ( size_ != 0 )
    {
        if ( auto code = logError( cudaMalloc( ( void** )&data_, size_ * sizeof( T ) ), __FILE__ , __LINE__ ) )
            return code;
    }
    return cudaSuccess;
}

template<typename T>
template<typename U>
cudaError_t DynamicArray<T>::toVector( std::vector<U>& vec ) const
{
    static_assert ( sizeof( T ) == sizeof( U ) );
    vec.resize( size_ );
    return logError( cudaMemcpy( vec.data(), data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost ), __FILE__ , __LINE__ );
}

inline cudaError_t setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return cudaSuccess;
    return logError( cudaMemset( devArray.data(), 0, devArray.size() * sizeof( float ) ), __FILE__ , __LINE__ );
}

} // namespace Cuda

} // namespace MR
