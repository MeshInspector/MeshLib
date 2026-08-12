#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma clang diagnostic ignored "-Wdeprecated" // covers redundant out-of-line constexpr static definitions in older boost
#endif

#include <boost/multiprecision/cpp_int.hpp>

#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace MR
{

/// \defgroup HighPrecisionGroup High Precision
/// \ingroup MathGroup
/// \{

using Int128 = boost::multiprecision::int128_t;
using Int256 = boost::multiprecision::int256_t;
using Int512 = boost::multiprecision::int512_t;
using Int1024 = boost::multiprecision::int1024_t;

using Vector2i128 = Vector2<Int128>;
using Vector3i128 = Vector3<Int128>;

// no bindings since no operator << and no sqrt for FastInt128
#if !defined MR_PARSING_FOR_ANY_BINDINGS && !defined MR_COMPILING_ANY_BINDINGS
using Vector2i128fast = Vector2<FastInt128>;
using Vector3i128fast = Vector3<FastInt128>;
#endif

using Vector2i256 = Vector2<Int256>;
using Vector3i256 = Vector3<Int256>;

using Vector3i512 = Vector3<Int512>;

/// the same value as a boost integer of the type B, which unlike FastInt supports division,
/// square root and stream output; B must be able to represent the value, and cpp_int always can
template <typename B, int nBits>
[[nodiscard]] MR_BIND_IGNORE B toBoostInt( const FastInt<nBits> & v )
{
    const bool neg = v.sign() < 0;
    auto w = v.w;
    if ( neg ) // the magnitude, so that 2^nBits is never needed in B
    {
        std::uint64_t borrow = 0;
        for ( auto & x : w )
            x = detail::subBorrow64( 0, x, borrow );
    }
    B res = 0;
    for ( int i = FastInt<nBits>::numWords - 1; i >= 0; --i )
    {
        res <<= 64;
        res |= B( w[i] );
    }
    return neg ? -res : res;
}


/// \}

} // namespace MR
