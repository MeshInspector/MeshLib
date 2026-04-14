#pragma once

#include "MRMesh/MRMeshFwd.h"
#include "MRPch/MRHashMap.h"

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
