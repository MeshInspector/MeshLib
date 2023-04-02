#include "MRCudaBasic.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
void DynamicArray<T>::fromVector( const std::vector<T>& vec )
{
    resize( vec.size() );
    cudaMemcpy( data_, vec.data(), size_ * sizeof( T ), cudaMemcpyHostToDevice );
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
void DynamicArray<T>::toVector( std::vector<T>& vec ) const
{
    vec.resize( size_ );
    cudaMemcpy( vec.data(), data_, size_ * sizeof( T ), cudaMemcpyDeviceToHost );
}

void setToZero( DynamicArrayF& devArray )
{
    if ( devArray.size() == 0 )
        return;
    cudaMemset( devArray.data(), 0, devArray.size() * sizeof( float ) );
}

bool isCudaAvailable()
{
    int n;
    cudaError err = cudaGetDeviceCount( &n );
    if ( err != cudaError::cudaSuccess )
        return false;
    return n > 0;
}

size_t getCudaAvailableMemory()
{
    if ( !isCudaAvailable() )
        return 0;
    cudaSetDevice( 0 );
    size_t memFree = 0, memTot = 0;
    cudaMemGetInfo( &memFree, &memTot );
    // minus extra 128 MB
    return memFree - 128 * 1024 * 1024;
}

template class DynamicArray<uint16_t>;
template class DynamicArray<float>;

}

}