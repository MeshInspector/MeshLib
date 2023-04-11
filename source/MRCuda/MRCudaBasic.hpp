#pragma once
#include "cuda_runtime.h"
#include <assert.h>
#include <stdint.h>

namespace MR
{

namespace Cuda
{

template<typename T>
DynamicArray<T>::DynamicArray( size_t size )
{
    resize( size );
}

template<typename T>
DynamicArray<T>::DynamicArray( const std::vector<T>& vec )
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
inline void DynamicArray<T>::fromVector( const std::vector<U>& vec )
{
    static_assert ( sizeof( T ) == sizeof( U ) ); 
    resize( vec.size() );
    cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
}


template <typename T>
void DynamicArray<T>::fromBytes( const uint8_t* data, size_t numBytes )
{
    assert( numBytes % sizeof( T ) == 0 );
    resize( numBytes / sizeof( T ) );
    cudaMemcpy( data_, data, numBytes, cudaMemcpyHostToDevice );
}

template <typename T>
void DynamicArray<T>::toBytes( uint8_t* data )
{
    cudaMemcpy( data, data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost );
}

template<typename T>
void DynamicArray<T>::resize( size_t size )
{
    if ( size == size_ )
        return;
    if ( size_ != 0 )
        cudaFree( data_ );

    size_ = size;
    if ( size_ != 0 )
        cudaMalloc( ( void** )&data_, size_ * sizeof( T ) );
}

template<typename T>
template<typename U>
void DynamicArray<T>::toVector( std::vector<U>& vec ) const
{
    static_assert ( sizeof( T ) == sizeof( U ) );
    vec.resize( size_ );
    cudaMemcpy( vec.data(), data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost );
}

inline void setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return;
    cudaMemset( devArray.data(), 0, devArray.size() * sizeof( float ) );
}

}
}
