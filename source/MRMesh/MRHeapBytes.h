#pragma once

#include "MRMeshFwd.h"
#include "MRMacros.h"
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
        size_t res = vec.capacity() * sizeof( T );
        for ( const T & t : vec )
            res += t.heapBytes();
        return res;
    }
    else
    {
        return vec.capacity() * sizeof( T );
    }
}

template<typename T, typename U>
[[nodiscard]] inline size_t heapBytes( const Vector<T, U>& vec )
{
    constexpr bool hasHeapBytes = requires( const T & t ) { t.heapBytes(); };
    if constexpr ( hasHeapBytes )
    {
        size_t res = vec.size() * sizeof( T );
        for ( const T & t : vec )
            res += t.heapBytes();
        return res;
    }
    else
    {
        return vec.size() * sizeof( T );
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

/// Needed for generic code, always returns zero.
/// The constraint is needed to avoid hard errors in C bindings when using MSVC STL when calling `heapBytes<SomeNonfuncType>(...)`.
template<typename T> MR_REQUIRES_IF_SUPPORTED( std::is_function_v<T> )
[[nodiscard]] inline size_t heapBytes( const std::function<T> & )
{
    return 0;
}

/// returns the amount of memory given HashMap occupies on heap
template<typename ...Ts>
[[nodiscard]] inline size_t heapBytes( const phmap::flat_hash_map<Ts...>& hashMap )
{
    // from parallel_hashmap/phmap.h:
    // The control state and slot array are stored contiguously in a shared heap
    // allocation. The layout of this allocation is: `capacity()` control bytes,
    // one sentinel control byte, `Group::kWidth - 1` cloned control bytes,
    // <possible padding>, `capacity()` slots
    const auto cap = hashMap.capacity();
    constexpr size_t kWidth = 16; // the usage of phmap::priv::Group::kWidth here will require inclusion of phmap.h
    return cap + kWidth + cap * sizeof( typename phmap::flat_hash_map<Ts...>::slot_type );
}

/// \}

} // namespace MR
