#pragma once

#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <array>
#include <compare>
#include <cstdint>
#include <type_traits>

namespace MR
{

namespace detail
{

/// returns a + b + carry modulo 2^64, and replaces carry with the carry-out (0 or 1)
[[nodiscard]] inline constexpr uint64_t addCarry64( uint64_t a, uint64_t b, uint64_t & carry ) noexcept
{
    const uint64_t s = a + b;
    const uint64_t r = s + carry;
    carry = uint64_t( s < a ) + uint64_t( r < s ); // at most one of the two additions can wrap
    return r;
}

/// returns a - b - borrow modulo 2^64, and replaces borrow with the borrow-out (0 or 1)
[[nodiscard]] inline constexpr uint64_t subBorrow64( uint64_t a, uint64_t b, uint64_t & borrow ) noexcept
{
    const uint64_t d = a - b;
    const uint64_t r = d - borrow;
    borrow = uint64_t( d > a ) + uint64_t( r > d ); // at most one of the two subtractions can wrap
    return r;
}

} // namespace detail

/// signed 256-bit integer with the operations necessary for exact geometric predicates:
/// the exact product of two FastInt128 (see mulExact below), addition, subtraction and comparison;
/// as FastInt128 it lacks conversion in double, sqrt-function and stream input/output;
/// there is no operator * on purpose: a product of two 256-bit values rarely fits in 256 bits
class MR_BIND_IGNORE FastInt256
{
public:
    /// two's complement representation: w[0] + 2^64 * w[1] + 2^128 * w[2] + 2^192 * int64_t( w[3] )
    std::array<uint64_t, 4> w = {};

    FastInt256() noexcept = default;

    /// sign-extends the given value, which must fit in 128 bits
    template <typename T>
    requires ( std::is_integral_v<T> || std::is_same_v<T, FastInt128> )
    constexpr FastInt256( T v ) noexcept
    {
        const FastInt128 x = FastInt128( v );
        w[0] = uint64_t( x );
        w[1] = uint64_t( x >> 64 );
        w[2] = w[3] = int64_t( w[1] ) < 0 ? ~uint64_t( 0 ) : 0;
    }

    /// -1, 0 or 1 if the value is negative, zero or positive respectively
    [[nodiscard]] constexpr int sign() const noexcept
    {
        if ( int64_t( w[3] ) < 0 )
            return -1;
        return ( w[0] | w[1] | w[2] | w[3] ) != 0 ? 1 : 0;
    }

    constexpr FastInt256 & operator +=( const FastInt256 & b ) noexcept
    {
        uint64_t carry = 0;
        for ( int i = 0; i < 4; ++i )
            w[i] = detail::addCarry64( w[i], b.w[i], carry );
        return *this;
    }

    constexpr FastInt256 & operator -=( const FastInt256 & b ) noexcept
    {
        uint64_t borrow = 0;
        for ( int i = 0; i < 4; ++i )
            w[i] = detail::subBorrow64( w[i], b.w[i], borrow );
        return *this;
    }

    [[nodiscard]] constexpr FastInt256 operator -() const noexcept
    {
        FastInt256 res;
        res -= *this;
        return res;
    }

    [[nodiscard]] friend constexpr FastInt256 operator +( FastInt256 a, const FastInt256 & b ) noexcept { a += b; return a; }
    [[nodiscard]] friend constexpr FastInt256 operator -( FastInt256 a, const FastInt256 & b ) noexcept { a -= b; return a; }

    [[nodiscard]] friend constexpr bool operator ==( const FastInt256 & a, const FastInt256 & b ) noexcept { return a.w == b.w; }

    [[nodiscard]] friend constexpr std::strong_ordering operator <=>( const FastInt256 & a, const FastInt256 & b ) noexcept
    {
        if ( const auto c = int64_t( a.w[3] ) <=> int64_t( b.w[3] ); c != std::strong_ordering::equal )
            return c;
        for ( int i = 2; i >= 0; --i )
            if ( const auto c = a.w[i] <=> b.w[i]; c != std::strong_ordering::equal )
                return c;
        return std::strong_ordering::equal;
    }
};

/// returns the exact product of two 128-bit integers, which in general does not fit in 128 bits;
/// operator * of FastInt128 keeps the lowest 128 bits of the product only
[[nodiscard]] inline constexpr FastInt256 mulExact( FastInt128 a, FastInt128 b ) noexcept
{
    const FastUInt128 ua( a ), ub( b );
    const uint64_t x[2] = { uint64_t( ua ), uint64_t( ua >> 64 ) };
    const uint64_t y[2] = { uint64_t( ub ), uint64_t( ub >> 64 ) };

    FastInt256 res;
    for ( int i = 0; i < 2; ++i )
    {
        uint64_t carry = 0;
        for ( int j = 0; j < 2; ++j )
        {
            // at most ( 2^64 - 1 )^2 + 2 * ( 2^64 - 1 ) < 2^128 here
            const FastUInt128 t = FastUInt128( x[i] ) * FastUInt128( y[j] ) + FastUInt128( res.w[i + j] ) + FastUInt128( carry );
            res.w[i + j] = uint64_t( t );
            carry = uint64_t( t >> 64 );
        }
        res.w[i + 2] = carry;
    }

    // the loop above has multiplied the arguments as if they were unsigned
    FastUInt128 high = ( FastUInt128( res.w[3] ) << 64 ) | FastUInt128( res.w[2] );
    if ( a < 0 )
        high -= ub;
    if ( b < 0 )
        high -= ua;
    res.w[2] = uint64_t( high );
    res.w[3] = uint64_t( high >> 64 );
    return res;
}

} // namespace MR
