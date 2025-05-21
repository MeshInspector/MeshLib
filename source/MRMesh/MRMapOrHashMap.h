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

    /// if this stores dense map then resizes it to denseTotalSize;
    /// if this stores hash map then sets its capacity to size()+hashAdditionalCapacity
    void resizeReserve( size_t denseTotalSize, size_t hashAdditionalCapacity );

    /// appends one element in the map,
    /// in case of dense map, key must be equal to vector.endId()
    void pushBack( K key, V val );

    /// executes given function for all pairs (key, value) with valid value for dense map
    template<typename F>
    void forEach( F && f ) const;

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
void MapOrHashMap<K,V>::resizeReserve( size_t denseTotalSize, size_t hashAdditionalCapacity )
{
    std::visit( overloaded{
        [denseTotalSize]( Dense& map ) { map.resize( denseTotalSize ); },
        [hashAdditionalCapacity]( Hash& hashMap ) { hashMap.reserve( hashMap.size() + hashAdditionalCapacity ); }
    }, var );
}

template <typename K, typename V>
void MapOrHashMap<K,V>::pushBack( K key, V val )
{
    std::visit( overloaded{
        [=]( Dense& map ) { assert( key == map.endId() ); map.push_back( val ); },
        [=]( Hash& hashMap ) { hashMap[key] = val; }
    }, var );
}

template <typename K, typename V>
template <typename F>
void MapOrHashMap<K,V>::forEach( F && f ) const
{
    std::visit( overloaded{
        [&f]( const Dense& map )
        {
            for ( K key( 0 ); key < map.size(); ++key )
                if ( auto val = map[key] )
                    f( key, val );
        },
        [&f]( const Hash& hashMap )
        {
            for ( const auto & [ key, val ] : hashMap )
                f( key, val );
        }
    }, var );
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
