#pragma once

#include "MRVector.h"
#include "MRphmap.h"
#include <variant>

namespace MR
{

/// stores a mapping from keys K to values V in one of two forms:
/// 1) as dense map (vector) preferable when there are few missing keys in a range [0, endKey)
/// 2) as hash map preferable when valid keys are a small subset of the range
template <typename K, typename V>
struct MapOrHashMap
{
    using Dense = Vector<V, K>;
    using Hash = HashMap<K, V>;
    // default construction will select dense map
    std::variant<Dense, Hash> var;

    [[nodiscard]] static MapOrHashMap createMap( size_t size = 0 );
    [[nodiscard]] static MapOrHashMap createHashMap( size_t capacity = 0 );

    void setMap( Dense && m ) { var = std::move( m ); }
    void setHashMap( Hash && m ) { var = std::move( m ); }

    [[nodiscard]]       Dense* getMap()       { return get_if<Dense>( &var ); }
    [[nodiscard]] const Dense* getMap() const { return get_if<Dense>( &var ); }

    [[nodiscard]]       Hash* getHashMap()       { return get_if<Hash>( &var ); }
    [[nodiscard]] const Hash* getHashMap() const { return get_if<Hash>( &var ); }

    void clear();
};

template <typename K, typename V>
inline MapOrHashMap<K,V> MapOrHashMap<K,V>::createMap( size_t size )
{
    MapOrHashMap<K,V> res;
    res.var = Dense( size );
    return res;
}

template <typename K, typename V>
inline MapOrHashMap<K,V> MapOrHashMap<K,V>::createHashMap( size_t capacity )
{
    MapOrHashMap<K,V> res;
    Hash hmap;
    if ( capacity > 0 )
        hmap.reserve( capacity );
    res.var = std::move( hmap );
    return res;
}

template <typename K, typename V>
void MapOrHashMap<K,V>::clear()
{
    std::visit( overloaded{
        []( Dense& map ) { map.clear(); },
        []( Hash& hashMap ) { hashMap.clear(); }
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
