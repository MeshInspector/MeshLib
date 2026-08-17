#pragma once

#include <limits>

namespace MR
{

constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();
constexpr int cQuietNanBits = __builtin_bit_cast( int, cQuietNan );

/// quickly tests whether given float is not-a-number
inline bool isNanFast( float f )
{
    return __builtin_bit_cast( int, f ) == cQuietNanBits;
}

} //namespace MR
