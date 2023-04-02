#pragma once
#include "exports.h"
#include <vector>

namespace MR
{

namespace Cuda
{
// This struct is present to simplify GPU memory control
template<typename T>
class MRCUDA_CLASS DynamicArray
{
public:
    DynamicArray() = default;
    // malloc given size on GPU
    MRCUDA_API DynamicArray( size_t size );
    // copy given vector to GPU
    MRCUDA_API DynamicArray( const std::vector<T>& vec );
    // free this array from GPU (if needed)
    MRCUDA_API ~DynamicArray();

    DynamicArray( const DynamicArray& ) = delete;
    DynamicArray( DynamicArray&& ) = delete;
    DynamicArray& operator=( DynamicArray&& ) = delete;
    DynamicArray& operator=( const DynamicArray& other ) = delete;

    // copy given vector to GPU (if this array was allocated with inconsistent size, free it and then malloc again)
    MRCUDA_API void fromVector( const std::vector<T>& vec );
    // copy this GPU array to given vector
    MRCUDA_API void toVector( std::vector<T>& vec ) const;
    // resize (free and malloc againg if size inconsistent) this GPU array (if size == 0 free it (if needed))
    MRCUDA_API void resize( size_t size );

    // pointer to GPU array
    T* data() { return data_; }
    // const pointer to GPU array
    const T* data() const { return data_; }
    // size of GPU array
    size_t size() const { return size_; }

private:
    T* data_{ nullptr };
    size_t size_{ 0 };
};

using DynamicArrayU16 = DynamicArray<uint16_t>;
using DynamicArrayF = DynamicArray<float>;

// Sets all float values of GPU array to zero
MRCUDA_API void setToZero( DynamicArrayF& devArray );

// Returns true if Cuda is present on this GPU
MRCUDA_API bool isCudaAvailable();

// Returns available GPU memory in bytes
MRCUDA_API size_t getCudaAvailableMemory();
}

}