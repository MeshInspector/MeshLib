#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include "MRResizeNoInit.h"
#include "MRPch/MRBindingMacros.h"
#include <cassert>

namespace MR
{

/**
 * \brief std::vector<T>-like container that requires specific indexing type,
 * \tparam T type of stored elements
 * \tparam I type of index (shall be convertible to size_t)
 * \ingroup BasicGroup
 */
template <typename T, typename I>
class Vector
{
public:
    using value_type = typename std::vector<T>::value_type;
    using reference = typename std::vector<T>::reference;
    using const_reference = typename std::vector<T>::const_reference;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    /// creates empty vector
    Vector() = default;

    /// creates a vector with \p size elements with default value
    MR_BIND_OVERLOAD_SUFFIX() explicit Vector( size_t size ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::default_initializable<T> ) : vec( size ) { }

    /// creates a vector with \p size elements with the given value
    MR_BIND_OVERLOAD_SUFFIX(fill) explicit Vector( size_t size, const T & val ) : vec( size, val ) { }

    /// moves data from the given std::vector<T>
    MR_BIND_OVERLOAD_SUFFIX(stdVector) Vector( std::vector<T> && v ) : vec( std::move( v ) ) { }

    template< class InputIt >
    Vector( InputIt first, InputIt last ) : vec( first, last ) { }

    Vector( std::initializer_list<T> init ) : vec( init ) { }

    [[nodiscard]] bool operator == ( const Vector & b ) const MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::equality_comparable<T> ) { return vec == b.vec; }
    [[nodiscard]] bool operator != ( const Vector & b ) const MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::equality_comparable<T> ) { return vec != b.vec; }

    void clear() { vec.clear(); }

    [[nodiscard]] bool empty() const { return vec.empty(); }

    [[nodiscard]] std::size_t size() const { return vec.size(); }

    void resize( size_t newSize ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> && std::default_initializable<T> ) { vec.resize( newSize ); }
    void resize( size_t newSize, const T & t ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> ) { vec.resize( newSize, t ); }

    // resizes the vector skipping initialization of its elements (more precisely initializing them using ( noInit ) constructor )
    void resizeNoInit( size_t targetSize ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::constructible_from<T, NoInit> ) { MR::resizeNoInit( vec, targetSize ); }

    [[nodiscard]] std::size_t capacity() const { return vec.capacity(); }

    void reserve( size_t capacity ) { vec.reserve( capacity ); }

    void shrink_to_fit() { vec.shrink_to_fit(); }

    [[nodiscard]] const_reference operator[]( I i ) const MR_LIFETIMEBOUND
    {
        assert( i < vec.size() );
        return vec[i];
    }
    [[nodiscard]] reference operator[]( I i ) MR_LIFETIMEBOUND
    {
        assert( i < vec.size() );
        return vec[i];
    }

    /// same as resize(newSize), which already grows the capacity geometrically
    [[deprecated( "use resize() instead" )]]
    void resizeWithReserve( size_t newSize ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::default_initializable<T> )
    {
        vec.resize( newSize );
    }

    /// same as resize(newSize, value), which already grows the capacity geometrically
    [[deprecated( "use resize() instead" )]]
    void resizeWithReserve( size_t newSize, const T & value ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> )
    {
        vec.resize( newSize, value );
    }

    /// sets elements [pos, pos+len) to the given value, adjusting the size of the vector to include new elements;
    /// the elements in between old size and \p pos are also set to \p val (for faster implementation)
    void autoResizeSet( I pos, size_t len, T val ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> && std::is_copy_assignable_v<T> )
    {
        assert( pos );
        const size_t p = pos;
        if ( const auto sz = size(); p + len > sz )
        {
            // add new elements with the given value
            vec.resize( p + len, val );
            if ( p >= sz )
                return;
            // the number of the elements existing before function call to be changed
            len = sz - p;
        }
        // change the value of the elements existing before function call
        for ( size_t i = 0; i < len; ++i )
            vec[ p + i ] = val;
    }

    /// sets the element #i to the given value, adjusting the size of the vector to include new element
    void autoResizeSet( I i, T val ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::is_copy_assignable_v<T> )
    {
        autoResizeSet( i, 1, val );
    }

    /// this accessor automatically adjusts the size of the vector
    [[nodiscard]] reference autoResizeAt( I i ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::default_initializable<T> )
    {
        if ( i + 1 > size() )
            vec.resize( i + 1 );
        return vec[i];
    }

    void push_back( const T & t MR_LIFETIME_CAPTURE_BY_NESTED(this) ) { vec.push_back( t ); }
    void push_back( T && t MR_LIFETIME_CAPTURE_BY_NESTED(this) ) { vec.push_back( std::move( t ) ); }
    void pop_back() { vec.pop_back(); }

    /// inserts elements from range [first, last) before \p pos
    template<class InputIt>
    iterator insert( const_iterator pos, InputIt first, InputIt last ) { return vec.insert( pos, first, last ); }

    template<typename... Args> MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::constructible_from<T, Args &&...> )
    T& emplace_back( Args&&... args ) { return vec.emplace_back( std::forward<Args>(args)... ); }

    [[nodiscard]] const_reference front() const MR_LIFETIMEBOUND { return vec.front(); }
    [[nodiscard]]       reference front()       MR_LIFETIMEBOUND { return vec.front(); }
    [[nodiscard]] const_reference  back() const MR_LIFETIMEBOUND { return vec.back(); }
    [[nodiscard]]       reference  back()       MR_LIFETIMEBOUND { return vec.back(); }

    /// returns the identifier of the first element
    [[nodiscard]] I beginId() const { return I( size_t(0) ); }

    /// returns the identifier of the back() element
    [[nodiscard]] I backId() const { assert( !vec.empty() ); return I( vec.size() - 1 ); }

    /// returns backId() + 1
    [[nodiscard]] I endId() const { return I( vec.size() ); }

    [[nodiscard]] T* data() MR_LIFETIMEBOUND { return vec.data(); }
    [[nodiscard]] const T* data() const MR_LIFETIMEBOUND { return vec.data(); }

    void swap( Vector & b ) { vec.swap( b.vec ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() * sizeof(T); }

    [[nodiscard]] MR_BIND_IGNORE_PY friend auto begin( const Vector<T, I> & a ) { return a.vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE_PY friend auto begin( Vector<T, I> & a ) { return a.vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE_PY friend auto end( const Vector<T, I> & a ) { return a.vec.end(); }
    [[nodiscard]] MR_BIND_IGNORE_PY friend auto end( Vector<T, I> & a ) { return a.vec.end(); }

    /// The user can directly manipulate the vector, they can't break anything anyway.
    std::vector<T> vec;

#if defined( MR_PARSING_FOR_ANY_BINDINGS ) || defined( MR_COMPILING_ANY_BINDINGS )
    static_assert( sizeof(T) > 0 );
#endif
};

/// given some std::vector and a key, returns the value associated with the key, or default value if key is invalid or outside the std::vector
template <typename T>
[[nodiscard]] inline T getAt( const std::vector<T>& a, size_t id, T def = {} )
{
    return  id < a.size() ? a[id] : def;
}

/// given some Vector and a key, returns the value associated with the key, or default value if key is invalid or outside the Vector
template <typename T, typename I>
[[nodiscard]] inline T getAt( const Vector<T, I> & a, I id, T def = {} )
{
    return getAt( a.vec, size_t( id ), def );
}

} // namespace MR
