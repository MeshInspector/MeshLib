#pragma once

#include "MRMacros.h"
#include "MRMeshFwd.h"
#include "MRAllocator.h"
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
    using StdVector = std::vector<T, Allocator<T>>;
    using value_type = typename StdVector::value_type;
    using reference = typename StdVector::reference;
    using const_reference = typename StdVector::const_reference;
    using size_type = typename StdVector::size_type;
    using difference_type = typename StdVector::difference_type;

    using iterator MR_BIND_IGNORE = typename StdVector::iterator;
    using const_iterator MR_BIND_IGNORE = typename StdVector::const_iterator;
    using reverse_iterator MR_BIND_IGNORE = typename StdVector::reverse_iterator;
    using const_reverse_iterator MR_BIND_IGNORE = typename StdVector::const_reverse_iterator;
    using pointer = typename StdVector::pointer;
    using const_pointer = typename StdVector::const_pointer;

    /// creates empty vector
    Vector() = default;

    /// creates a vector with \p size elements with default value
    MR_BIND_OVERLOAD_SUFFIX() explicit Vector( size_t size ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::default_initializable<T> ) : vec( size ) { }

    /// creates a vector with \p size elements with the given value
    MR_BIND_OVERLOAD_SUFFIX(fill) explicit Vector( size_t size, const T & val ) : vec( size, val ) { }

    template< class InputIt >
    Vector( InputIt first, InputIt last ) : vec( first, last ) { }

    Vector( std::initializer_list<T> init ) : vec( init ) { }

    /// replaces the contents with \p count copies of \p val
    void assign( size_t count, const T & val MR_LIFETIME_CAPTURE_BY_NESTED(this) ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::is_copy_assignable_v<T> ) { vec.assign( count, val ); }

    /// replaces the contents with elements from range [first, last)
    template< class InputIt >
    void assign( InputIt first, InputIt last ) { vec.assign( first, last ); }

    void assign( std::initializer_list<T> init ) { vec.assign( init ); }

    [[nodiscard]] bool operator == ( const Vector & b ) const MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::equality_comparable<T> ) { return vec == b.vec; }
    [[nodiscard]] bool operator != ( const Vector & b ) const MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::equality_comparable<T> ) { return vec != b.vec; }

    void clear() { vec.clear(); }

    [[nodiscard]] bool empty() const { return vec.empty(); }

    [[nodiscard]] std::size_t size() const { return vec.size(); }

    [[nodiscard]] std::size_t max_size() const { return vec.max_size(); }

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

    /// inserts \p t before \p pos
    MR_BIND_IGNORE iterator insert( const_iterator pos, const T & t MR_LIFETIME_CAPTURE_BY_NESTED(this) ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::copy_constructible<T> && std::movable<T> ) { return vec.insert( pos, t ); }
    MR_BIND_IGNORE iterator insert( const_iterator pos, T && t MR_LIFETIME_CAPTURE_BY_NESTED(this) ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> ) { return vec.insert( pos, std::move( t ) ); }

    /// inserts \p count copies of \p t before \p pos
    MR_BIND_IGNORE iterator insert( const_iterator pos, size_t count, const T & t MR_LIFETIME_CAPTURE_BY_NESTED(this) ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::copy_constructible<T> && std::movable<T> ) { return vec.insert( pos, count, t ); }

    /// inserts elements from range [first, last) before \p pos
    template<class InputIt>
    iterator insert( const_iterator pos, InputIt first, InputIt last ) { return vec.insert( pos, first, last ); }

    MR_BIND_IGNORE iterator insert( const_iterator pos, std::initializer_list<T> init ) { return vec.insert( pos, init ); }

    /// constructs an element in-place before \p pos
    template<typename... Args> MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::constructible_from<T, Args &&...> && std::movable<T> )
    iterator emplace( const_iterator pos, Args&&... args ) { return vec.emplace( pos, std::forward<Args>(args)... ); }

    /// removes the element at \p pos
    MR_BIND_IGNORE iterator erase( const_iterator pos ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> ) { return vec.erase( pos ); }

    /// removes the elements in range [first, last)
    MR_BIND_IGNORE iterator erase( const_iterator first, const_iterator last ) MR_REQUIRES_IF_SUPPORTED( sizeof(T)>0 && std::movable<T> ) { return vec.erase( first, last ); }

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

    [[nodiscard]] MR_BIND_IGNORE iterator begin() MR_LIFETIMEBOUND { return vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE const_iterator begin() const MR_LIFETIMEBOUND { return vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE const_iterator cbegin() const MR_LIFETIMEBOUND { return vec.cbegin(); }
    [[nodiscard]] MR_BIND_IGNORE iterator end() MR_LIFETIMEBOUND { return vec.end(); }
    [[nodiscard]] MR_BIND_IGNORE const_iterator end() const MR_LIFETIMEBOUND { return vec.end(); }
    [[nodiscard]] MR_BIND_IGNORE const_iterator cend() const MR_LIFETIMEBOUND { return vec.cend(); }
    [[nodiscard]] MR_BIND_IGNORE reverse_iterator rbegin() MR_LIFETIMEBOUND { return vec.rbegin(); }
    [[nodiscard]] MR_BIND_IGNORE const_reverse_iterator rbegin() const MR_LIFETIMEBOUND { return vec.rbegin(); }
    [[nodiscard]] MR_BIND_IGNORE const_reverse_iterator crbegin() const MR_LIFETIMEBOUND { return vec.crbegin(); }
    [[nodiscard]] MR_BIND_IGNORE reverse_iterator rend() MR_LIFETIMEBOUND { return vec.rend(); }
    [[nodiscard]] MR_BIND_IGNORE const_reverse_iterator rend() const MR_LIFETIMEBOUND { return vec.rend(); }
    [[nodiscard]] MR_BIND_IGNORE const_reverse_iterator crend() const MR_LIFETIMEBOUND { return vec.crend(); }

    void swap( Vector & b ) { vec.swap( b.vec ); }

    /// returns the amount of memory this object occupies on heap
    [[nodiscard]] size_t heapBytes() const { return capacity() * sizeof(T); }

    [[nodiscard]] MR_BIND_IGNORE friend const_iterator begin( const Vector<T, I> & a ) { return a.vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE friend iterator begin( Vector<T, I> & a ) { return a.vec.begin(); }
    [[nodiscard]] MR_BIND_IGNORE friend const_iterator end( const Vector<T, I> & a ) { return a.vec.end(); }
    [[nodiscard]] MR_BIND_IGNORE friend iterator end( Vector<T, I> & a ) { return a.vec.end(); }

    /// removes all elements equal to \p value, returns the number of removed elements
    template <typename U>
    friend size_t erase( Vector<T, I> & a, const U & value ) { return std::erase( a.vec, value ); }

    /// removes all elements satisfying \p pred, returns the number of removed elements
    template <typename Pred>
    friend size_t erase_if( Vector<T, I> & a, Pred pred ) { return std::erase_if( a.vec, pred ); }

    /// The user can directly manipulate the vector, they can't break anything anyway.
    StdVector vec;

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
    return size_t( id ) < a.size() ? a[id] : def;
}

} // namespace MR
