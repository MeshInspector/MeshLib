#pragma once

#include "MRMeshFwd.h"
#include <vector>
#include <memory>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// returns the amount of memory given vector occupies on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::vector<T> & vec )
{
    constexpr bool hasHeapBytes = requires( const T& t ) { t.heapBytes(); };
    if constexpr ( hasHeapBytes )
    {
        size_t res = 0;
        for ( const T & t : vec )
            res += t.heapBytes();
        return res;
    }
    else
    {
        return vec.capacity() * sizeof( T );
    }
}

/// returns the amount of memory this smart pointer and its pointed object own together on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::unique_ptr<T> & ptr )
{
    if ( !ptr )
        return 0;
    return sizeof( T ) + ptr->heapBytes();
}

/// returns the amount of memory this smart pointer and its pointed object own together on heap
template<typename T>
[[nodiscard]] inline size_t heapBytes( const std::shared_ptr<T> & ptr )
{
    if ( !ptr )
        return 0;
    return sizeof( T ) + ptr->heapBytes();
}

/// \}

} // namespace MR
