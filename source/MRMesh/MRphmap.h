#pragma once

#include "MRHash.h"

#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#endif
#include <parallel_hashmap/phmap.h>
#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

// given some hash map and a key, returns the value associated with the key, or default value if key is invalid or not found in the map
template <typename K, typename V>
inline V getAt( const MR::HashMap<K, V> & hmap, K key )
{
    if ( key )
    {
        if ( auto it = hmap.find( key ); it != hmap.end() )
            return it->second;
    }
    return {};
}
