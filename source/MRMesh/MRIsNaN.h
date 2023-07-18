#pragma once

#include <bit>
#if defined( __GNUC__ ) && __GNUC__ < 11
#include <cmath>
#else
#include <limits>
#endif

namespace MR
{

constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();
#if !defined( __GNUC__ ) || __GNUC__ >= 11
constexpr int cQuietNanBits = std::bit_cast< int >( cQuietNan );
#endif

/// quickly tests whether given float is not-a-number
inline bool isNanFast( float f )
{
#if defined( __GNUC__ ) && __GNUC__ < 11
    return std::isnan( f );
#else
    return std::bit_cast< int >( f ) == cQuietNanBits;
#endif
}

} //namespace MR
