#pragma once

#include <limits>

#if __cpp_lib_bit_cast >= 201806L
#include <bit>
#else
#include <cmath>
#endif

namespace MR
{

constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();
#if __cpp_lib_bit_cast >= 201806L
constexpr int cQuietNanBits = std::bit_cast< int >( cQuietNan );
#endif

/// quickly tests whether given float is not-a-number
inline bool isNanFast( float f )
{
#if __cpp_lib_bit_cast >= 201806L
    return std::bit_cast< int >( f ) == cQuietNanBits;
#else
    return std::isnan( f );
#endif
}

} //namespace MR
