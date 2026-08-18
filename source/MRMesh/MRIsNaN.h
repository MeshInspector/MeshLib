#pragma once

#include "MRBitCast.h"
#include <limits>

namespace MR
{

constexpr float cQuietNan = std::numeric_limits<float>::quiet_NaN();
constexpr int cQuietNanBits = MR::bit_cast<int>( cQuietNan );

/// quickly tests whether given float is not-a-number
inline bool isNanFast( float f )
{
    return MR::bit_cast<int>( f ) == cQuietNanBits;
}

} //namespace MR
