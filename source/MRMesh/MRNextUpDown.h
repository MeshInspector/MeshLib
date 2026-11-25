#pragma once

#include "MRBitCast.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace MR
{

namespace detail
{

/// +1 for positive x and -1 (all bits set) for negative x, as an unsigned integer of the same size as x
template <typename T>
[[nodiscard]] inline auto signStep( T x )
{
    using U = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
    using S = std::make_signed_t<U>;
    return U( U( S( bit_cast<U>( x ) ) >> ( 8 * sizeof( U ) - 1 ) ) | 1 );
}

/// whether 0 < |x| < max: then the closest values around x differ from it by one in the bits; one unsigned comparison of the bits
template <typename T>
[[nodiscard]] inline bool nonzeroBelowMax( T x )
{
    using U = decltype( signStep( x ) );
    constexpr U absMask = U( -1 ) >> 1;
    return U( ( bit_cast<U>( x ) & absMask ) - 1 ) < U( bit_cast<U>( std::numeric_limits<T>::max() ) - 1 );
}

} //namespace detail

/// returns the closest representable value greater than x,
/// the same as std::nextafter( x, std::numeric_limits<T>::max() ) but inline for finite nonzero floats and doubles
template <typename T>
[[nodiscard]] inline T nextUp( T x )
{
    if constexpr ( std::is_same_v<T, float> || std::is_same_v<T, double> )
    {
        // without a branch on the sign of x, which is often unpredictable
        if ( detail::nonzeroBelowMax( x ) )
            return bit_cast<T>( bit_cast<decltype( detail::signStep( x ) )>( x ) + detail::signStep( x ) );
    }
    return std::nextafter( x, std::numeric_limits<T>::max() );
}

/// returns the closest representable value less than x,
/// the same as std::nextafter( x, std::numeric_limits<T>::lowest() ) but inline for finite nonzero floats and doubles
template <typename T>
[[nodiscard]] inline T nextDown( T x )
{
    if constexpr ( std::is_same_v<T, float> || std::is_same_v<T, double> )
    {
        if ( detail::nonzeroBelowMax( x ) )
            return bit_cast<T>( bit_cast<decltype( detail::signStep( x ) )>( x ) - detail::signStep( x ) );
    }
    return std::nextafter( x, std::numeric_limits<T>::lowest() );
}

} //namespace MR
