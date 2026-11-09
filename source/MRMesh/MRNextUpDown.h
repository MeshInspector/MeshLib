#pragma once

#include "MRBitCast.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace MR
{

/// returns the closest representable value greater than x,
/// the same as std::nextafter( x, std::numeric_limits<T>::max() ) but inline for finite nonzero floats and doubles
template <typename T>
[[nodiscard]] inline T nextUp( T x )
{
    if constexpr ( std::is_same_v<T, float> || std::is_same_v<T, double> )
    {
        using U = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
        if ( x > 0 && x < std::numeric_limits<T>::max() )
            return bit_cast<T>( bit_cast<U>( x ) + 1 );
        if ( x < 0 && x >= std::numeric_limits<T>::lowest() )
            return bit_cast<T>( bit_cast<U>( x ) - 1 );
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
        using U = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
        if ( x > 0 && x <= std::numeric_limits<T>::max() )
            return bit_cast<T>( bit_cast<U>( x ) - 1 );
        if ( x < 0 && x > std::numeric_limits<T>::lowest() )
            return bit_cast<T>( bit_cast<U>( x ) + 1 );
    }
    return std::nextafter( x, std::numeric_limits<T>::lowest() );
}

} //namespace MR
