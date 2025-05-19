#pragma once

#include "MRVector.h"
#include "MRphmap.h"
#include <variant>

namespace MR
{

template <typename K, typename V>
struct MapOrHashMap
{
    using Map = Vector<V, K>;
    using HashMap = HashMap<K, V>;
    // default construction will select dense map
    std::variant<Map, HashMap> var;

    [[nodiscard]] static MapOrHashMap createMap( size_t size = 0 );
    [[nodiscard]] static MapOrHashMap createHashMap( size_t capacity = 0 );

    [[nodiscard]] const Map* getMap() const { return get_if<Map>( &var ); }
    [[nodiscard]] const HashMap* getHashMap() const { return get_if<HashMap>( &var ); }

    void clear();
};

template <typename K, typename V>
inline MapOrHashMap<K,V> MapOrHashMap<K,V>::createMap( size_t size )
{
    MapOrHashMap<K,V> res;
    res.var = Map( size );
    return res;
}

template <typename K, typename V>
inline MapOrHashMap<K,V> MapOrHashMap<K,V>::createHashMap( size_t capacity )
{
    MapOrHashMap<K,V> res;
    HashMap hmap;
    if ( capacity > 0 )
        hmap.reserve( capacity );
    res.var = std::move( hmap );
    return res;
}

template <typename K, typename V>
void MapOrHashMap<K,V>::clear()
{
    std::visit( overloaded{
        []( Map& map ) { map.clear(); },
        []( HashMap& hashMap ) { hashMap.clear(); }
    }, var );
}

template <typename K, typename V>
[[nodiscard]] inline V getAt( const MapOrHashMap<K, V> & m, K key, V def = {} )
{
    return std::visit( overloaded{
        [key, def]( const Vector<V, K>& map ) { return getAt( map, key, def ); },
        [key, def]( const HashMap<K, V>& hashMap ) { return getAt( hashMap, key, def ); }
    }, m.var );
}

template <typename K, typename V>
inline void setAt( MapOrHashMap<K, V> & m, K key, V val )
{
    std::visit( overloaded{
        [key, val]( Vector<V, K>& map ) { map[key] = val; },
        [key, val]( HashMap<K, V>& hashMap ) { hashMap[key] = val; }
    }, m.var );
}

} //namespace MR
