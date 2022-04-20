#pragma once

#include "MRMeshFwd.h"
#include <vector>

namespace MR
{

// returns the amount of memory given vector occupies on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::vector<T> & vec )
{
    return vec.capacity() * sizeof( typename std::vector<T>::value_type );
}

} //namespace MR
