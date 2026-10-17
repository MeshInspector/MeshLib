#pragma once

#include "MRBitCast.h"
#include <cmath>
#include <cstdint>

namespace MR
{

// MSVC (x64 and arm64) and GCC on x64 compile std::round into a library call,
// while GCC and Clang on arm64 compile it into a single fcvtas instruction
#if ( defined( _MSC_VER ) && !defined( __clang__ ) ) || defined( __x86_64__ ) || defined( _M_X64 )
#define MR_ROUND_TO_INT_NO_LIBRARY_CALL
#endif

/// the same as (int)std::round( x ) for any x in int range (halves are rounded away from zero), but without a library call on any compiler
[[nodiscard]] inline int roundToInt( double x ) noexcept
{
#ifdef MR_ROUND_TO_INT_NO_LIBRARY_CALL
    // add the largest double below 0.5 with the sign of x, then truncate
    const double half = bit_cast<double>( bit_cast<std::uint64_t>( 0.49999999999999994 ) | ( bit_cast<std::uint64_t>( x ) & 0x8000000000000000ull ) );
    return int( x + half );
#else
    return int( std::round( x ) );
#endif
}

/// the same as (int)std::round( x ) for any x in int range (halves are rounded away from zero), but without a library call on any compiler
[[nodiscard]] inline int roundToInt( float x ) noexcept
{
#ifdef MR_ROUND_TO_INT_NO_LIBRARY_CALL
    // add the largest float below 0.5 with the sign of x, then truncate
    const float half = bit_cast<float>( bit_cast<std::uint32_t>( 0.49999997f ) | ( bit_cast<std::uint32_t>( x ) & 0x80000000u ) );
    return int( x + half );
#else
    return int( std::round( x ) );
#endif
}

} //namespace MR
