#pragma once

#include "MRMeshFwd.h"
#include <vector>
#include <memory>

namespace MR
{

// returns the amount of memory given vector occupies on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::vector<T> & vec )
{
    return vec.capacity() * sizeof( typename std::vector<T>::value_type );
}

// returns the amount of memory this smart pointer and its pointed object own together on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::unique_ptr<T> & ptr )
{
    if ( !ptr )
        return 0;
    return sizeof( T ) + ptr->heapBytes();
}

// returns the amount of memory this smart pointer and its pointed object own together on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::shared_ptr<T> & ptr )
{
    if ( !ptr )
        return 0;
    return sizeof( T ) + ptr->heapBytes();
}

} //namespace MR
