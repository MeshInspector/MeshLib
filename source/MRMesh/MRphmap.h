#pragma once

#include "MRMesh/MRMeshFwd.h"

#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-builtins"
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wshift-count-overflow"
#endif
#include <parallel_hashmap/phmap.h>
#if defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

namespace MR
{

// given some hash map and a key, returns the value associated with the key, or default value if key is invalid or not found in the map
template <typename K, typename V>
inline V getAt( const HashMap<K, V> & hmap, K key, V def = {} )
{
    if ( key )
    {
        if ( auto it = hmap.find( key ); it != hmap.end() )
            return it->second;
    }
    return def;
}

} //namespace MR
