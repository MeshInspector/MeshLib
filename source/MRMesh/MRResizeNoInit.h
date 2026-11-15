#pragma once

#include "MRMeshFwd.h"
#include "MRMacros.h"
#include <compare>
#include <concepts>
#include <cstddef>
#include <iterator>

namespace MR
{

namespace detail
{

/// random-access iterator over a range of not-initialized T for std::vector::insert
template <typename T>
struct NoInitIterator
{
    /// T( ref ) picks T( NoInit ) by the derived-to-base conversion, so the element is constructed in place;
    /// the conversion to const T& is only for `*pos = *it` of an insert in the middle, which compiles but never runs here
    struct Ref : NoInit
    {
        operator const T&() const { static const T t( noInit ); return t; }
    };

    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = Ref;

    difference_type i = 0;

    Ref operator*() const { return {}; }
    Ref operator[]( difference_type ) const { return {}; }
    NoInitIterator& operator++() { ++i; return *this; }
    NoInitIterator operator++( int ) { auto t = *this; ++i; return t; }
    NoInitIterator& operator--() { --i; return *this; }
    NoInitIterator operator--( int ) { auto t = *this; --i; return t; }
    NoInitIterator& operator+=( difference_type d ) { i += d; return *this; }
    NoInitIterator& operator-=( difference_type d ) { i -= d; return *this; }
    friend NoInitIterator operator+( NoInitIterator a, difference_type d ) { return a += d; }
    friend NoInitIterator operator+( difference_type d, NoInitIterator a ) { return a += d; }
    friend NoInitIterator operator-( NoInitIterator a, difference_type d ) { return a -= d; }
    friend difference_type operator-( NoInitIterator a, NoInitIterator b ) { return a.i - b.i; }
    friend bool operator==( NoInitIterator a, NoInitIterator b ) { return a.i == b.i; }
    friend auto operator<=>( NoInitIterator a, NoInitIterator b ) { return a.i <=> b.i; }
};

} //namespace detail

/// how resizeNoInit reserves the memory when the capacity is short
enum class ReserveMode
{
    Exact, ///< exactly the target size: for a vector resized once
    Grow   ///< the vector grows the capacity geometrically by its own policy, as push_back does, so repeated small growths copy the elements only O(log) times
};

/// resizes the vector skipping initialization of its elements (more precisely initializing them using ( noInit ) constructor );
/// this is much faster than ordinary resize(), because the memory of vector's elements is not accessed,
/// and the optimizer removes the loop over the no-op constructors, so the time does not depend on the number of new elements
template <typename T, typename A>
void resizeNoInit( std::vector<T, A> & vec, size_t targetSize, ReserveMode mode = ReserveMode::Exact ) MR_REQUIRES_IF_SUPPORTED( sizeof( T ) > 0 && std::constructible_from<T, NoInit> )
{
    if ( targetSize <= vec.size() )
    {
        vec.resize( targetSize );
        return;
    }
    if ( mode == ReserveMode::Exact )
        vec.reserve( targetSize );
    // inserting a counted range lets the vector apply its own capacity growth
    vec.insert( vec.end(), detail::NoInitIterator<T>{}, detail::NoInitIterator<T>{ std::ptrdiff_t( targetSize - vec.size() ) } );
}

} //namespace MR
