#pragma once

#include "MRMeshFwd.h"
#include <cassert>
#include <vector>

namespace MR
{
 
// std::vector<T>-like container that requires specific indexing type,
// T - type of stored elements
// I - type of index (shall be convertible to size_t)
template <typename T, typename I>
class Vector
{
public:
    using reference = typename std::vector<T>::reference;
    using const_reference = typename std::vector<T>::const_reference;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    Vector() = default;
    explicit Vector( size_t size ) : vec_( size ) { }
    explicit Vector( size_t size, const T & val ) : vec_( size, val ) { }
    Vector( std::vector<T> && vec ) : vec_( std::move( vec ) ) { }

    bool operator == ( const Vector & b ) const { return vec_ == b.vec_; }
    bool operator != ( const Vector & b ) const { return vec_ != b.vec_; }

    void clear() { vec_.clear(); }
    bool empty() const { return vec_.empty(); }

    auto size() const { return vec_.size(); }

    void resize( size_t newSize ) { vec_.resize( newSize ); }
    void resize( size_t newSize, const T & t ) { vec_.resize( newSize, t ); }

    auto capacity() const { return vec_.capacity(); }
    void reserve( size_t capacity ) { vec_.reserve( capacity ); }

    const_reference operator[]( I i ) const
    {
        assert( i < vec_.size() );
        return vec_[i];
    }
    reference operator[]( I i )
    {
        assert( i < vec_.size() );
        return vec_[i];
    }

    // doubles reserved memory until resize(newSize) can be done without reallocation
    void resizeWithReserve( size_t newSize )
    {
        auto reserved = vec_.capacity();
        if ( reserved > 0 && newSize > reserved )
        {
            while ( newSize > reserved )
                reserved <<= 1;
            vec_.reserve( reserved );
        }
        vec_.resize( newSize );
    }

    // this accessor automatically adjusts the size of the vector
    reference autoResizeAt( I i )
    {
        if ( i + 1 > size() )
            resizeWithReserve( i + 1 );
        return vec_[i];
    }

    void push_back( const T & t ) { vec_.push_back( t ); }
    void push_back( T && t ) { vec_.push_back( std::move( t ) ); }
    void pop_back() { vec_.pop_back(); }

    template<typename... Args>
    void emplace_back( Args&&... args )   { vec_.emplace_back( std::forward<Args>(args)... ); }

    const_reference front() const { return vec_.front(); }
          reference front()       { return vec_.front(); }
    const_reference  back() const { return vec_.back(); }
          reference  back()       { return vec_.back(); }
    // returns the identifier of the back() element
    I backId() const { assert( !vec_.empty() ); return I{ vec_.size() - 1 }; }

    auto data() { return vec_.data(); }
    auto data() const { return vec_.data(); }

    void swap( Vector & b ) { vec_.swap( b.vec_ ); }

    // the user can directly manipulate the vector, anyway she cannot break anything
    std::vector<T> vec_;
};

template <typename T, typename I>
inline auto begin( const Vector<T, I> & a )
    { return a.vec_.begin(); }

template <typename T, typename I>
inline auto begin( Vector<T, I> & a )
    { return a.vec_.begin(); }

template <typename T, typename I>
inline auto end( const Vector<T, I> & a )
    { return a.vec_.end(); }

template <typename T, typename I>
inline auto end( Vector<T, I> & a )
    { return a.vec_.end(); }

} //namespace MR
