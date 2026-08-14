#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <compare>
#include <cstdint>
#include <type_traits>

namespace MR
{

/// \addtogroup HighPrecisionGroup
/// \{

namespace detail
{

/// the integer types that FastInt128 represents exactly, and which the classes below accept
template <typename T>
constexpr bool cFitsFastInt128 = std::is_integral_v<T> || std::is_same_v<T, FastInt128>;

// the two functions below are spelled via FastUInt128, where the carry-out is simply
// the higher word, as in mulWords below

/// returns a + b + carry modulo 2^64, and replaces carry with the carry-out (0 or 1)
[[nodiscard]] inline constexpr std::uint64_t addCarry64( std::uint64_t a, std::uint64_t b, std::uint64_t & carry ) noexcept
{
    const FastUInt128 t = FastUInt128( a ) + FastUInt128( b ) + FastUInt128( carry );
    carry = std::uint64_t( t >> 64 );
    return std::uint64_t( t );
}

/// returns a - b - borrow modulo 2^64, and replaces borrow with the borrow-out (0 or 1)
[[nodiscard]] inline constexpr std::uint64_t subBorrow64( std::uint64_t a, std::uint64_t b, std::uint64_t & borrow ) noexcept
{
    const FastUInt128 t = FastUInt128( a ) - FastUInt128( b ) - FastUInt128( borrow );
    borrow = std::uint64_t( t >> 64 ) & 1;
    return std::uint64_t( t );
}

/// all ones if the highest bit of the given word is set, and zeros otherwise
[[nodiscard]] inline constexpr std::uint64_t signWord( std::uint64_t hi ) noexcept
{
    return std::int64_t( hi ) < 0 ? ~std::uint64_t( 0 ) : 0;
}

/// the exact product of two two's-complement values given by their 64-bit words,
/// which always fits in the sum of their word counts; the only multiplication of this file
template <std::size_t n, std::size_t m> // std::size_t and not int, to be deducible from std::array
[[nodiscard]] constexpr std::array<std::uint64_t, n + m> mulWords(
    const std::array<std::uint64_t, n> & a, const std::array<std::uint64_t, m> & b ) noexcept
{
    std::array<std::uint64_t, n + m> res = {};
    for ( std::size_t i = 0; i < n; ++i ) // schoolbook multiplication of unsigned values
    {
        if ( a[i] == 0 )
            continue; // this row adds nothing: 0 * b[j] leaves every res[i + j] unchanged, and
                      // res[i + m] is still 0 (never written before this row), so the skipped
                      // res[i + m] = carry (carry stays 0 here) would be a no-op. Small-magnitude
                      // values keep their high words at 0, so this is the common fast path; negative
                      // operands are sign-extended to all-ones top words and are not skipped.
        std::uint64_t carry = 0;
        for ( std::size_t j = 0; j < m; ++j )
        {
            // at most ( 2^64 - 1 )^2 + 2 * ( 2^64 - 1 ) < 2^128 here
            const FastUInt128 t = FastUInt128( a[i] ) * FastUInt128( b[j] ) + FastUInt128( res[i + j] ) + FastUInt128( carry );
            res[i + j] = std::uint64_t( t );
            carry = std::uint64_t( t >> 64 );
        }
        res[i + m] = carry; // never written before, since the loop above stops at i + m - 1
    }

    // a negative argument was taken 2^(64*words) times larger than it is; both subtractions
    // below end exactly at the end of res, so the borrow never leaves it
    if ( std::int64_t( a[n - 1] ) < 0 )
    {
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < m; ++i )
            res[n + i] = subBorrow64( res[n + i], b[i], borrow );
    }
    if ( std::int64_t( b[m - 1] ) < 0 )
    {
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < n; ++i )
            res[m + i] = subBorrow64( res[m + i], a[i], borrow );
    }
    return res;
}

/// the number of bits a multiplier of FastInt occupies: 128 for FastInt128 and for an unsigned
/// 64-bit integer, both of which need two words, and 64 for anything signed or narrower
template <typename T>
constexpr int cMulBits = std::is_same_v<T, FastInt128> || ( std::is_unsigned_v<T> && sizeof( T ) >= 8 ) ? 128 : 64;

/// the words of a multiplier of FastInt, sign-extended if it is signed
template <typename T>
[[nodiscard]] constexpr auto mulWordsOf( T v ) noexcept
{
    if constexpr ( cMulBits<T> == 128 )
    {
        const FastInt128 x = FastInt128( v );
        return std::array{ std::uint64_t( x ), std::uint64_t( x >> 64 ) };
    }
    else
        return std::array{ std::uint64_t( std::int64_t( v ) ) };
}

/// the nearest double to the two's-complement value in the given words, least significant first;
/// see toDouble below for the guarantees, which this function alone provides for the whole family
template <std::size_t nWords>
[[nodiscard]] inline double doubleFromWords( std::array<std::uint64_t, nWords> w ) noexcept
{
    static_assert( nWords >= 1 );
    const bool neg = std::int64_t( w[nWords - 1] ) < 0;
    if ( neg )
    {
        // the magnitude, by negating in place; the smallest value negates into 2^(64*nWords-1),
        // which is not representable as a signed value here, but is as an unsigned magnitude
        std::uint64_t carry = 1;
        for ( std::size_t i = 0; i < nWords; ++i )
        {
            w[i] = ~w[i] + carry;
            carry = carry != 0 && w[i] == 0 ? 1 : 0;
        }
    }

    std::size_t k = nWords; // one past the highest non-zero word of the magnitude
    while ( k > 0 && w[k - 1] == 0 )
        --k;
    if ( k == 0 )
        return 0; // and not -0 for a negative zero, which two's complement has no representation of

    // the top 64 significant bits, with the highest one in bit 63: the word below k contributes
    // the bits shifted in from the right, and the shift by s cannot lose anything, because the
    // top s bits of w[k-1] are zero by the definition of s
    const int s = std::countl_zero( w[k - 1] );
    std::uint64_t top = w[k - 1] << s;
    if ( s > 0 && k >= 2 )
        top |= w[k - 2] >> ( 64 - s );

    // everything below those 64 bits, as a single sticky bit in the lowest one. That keeps the
    // conversion of top correctly rounded for the whole value: only 53 of its bits reach the
    // mantissa, so bit 0 sits well below the rounding position, and a non-zero tail there is
    // exactly what tells a tie (round to even) from a value strictly above it (round up)
    bool tail = k >= 2 && ( w[k - 2] << s ) != 0; // the bits of w[k-2] that top did not take, all of it if s is 0
    for ( std::size_t i = 0; i + 2 < k && !tail; ++i )
        tail = w[i] != 0; // the words entirely below top
    if ( tail )
        top |= 1;

    // std::uint64_t -> double is correctly rounded, and the scaling by a power of two is exact
    // unless it overflows, which yields the infinity of the right sign as intended
    const double res = std::ldexp( double( top ), int( 64 * ( k - 1 ) ) - s );
    return neg ? -res : res;
}

} // namespace detail

/// the nearest double to the given value: exact for a magnitude below 2^53, correctly rounded
/// otherwise (so with a relative error below 2^-53), and +-infinity past DBL_MAX. The bound is
/// load-bearing: the pre-filters that reject a case in double before evaluating it exactly are
/// only safe against a stated error of the conversion feeding them
[[nodiscard]] inline double toDouble( FastInt128 v ) noexcept
{
    const FastUInt128 u( v );
    // deliberately the same code as for FastInt below, and not the built-in conversion of
    // __int128_t, which MSVC's std::_Signed128 lacks: the value must not depend on the platform
    return detail::doubleFromWords( std::array{ std::uint64_t( u ), std::uint64_t( u >> 64 ) } );
}

/// signed integer of nBits bits, which must be a multiple of 64 and at least 192
/// (below that use FastInt128 with Int64Mul128 and Int128Mul256);
/// the product of two of them is exact, because it is twice as wide as the arguments;
/// as FastInt128 it lacks a sqrt-function and stream input/output
template <int nBits>
class MR_BIND_IGNORE FastInt
{
public:
    static_assert( nBits >= 192 && nBits % 64 == 0 );

    /// the number of words in the representation below
    static constexpr int numWords = nBits / 64;

    /// two's complement representation: w[0] + 2^64 * w[1] + ... + 2^(nBits-64) * std::int64_t( w[numWords-1] )
    std::array<std::uint64_t, numWords> w = {};

    FastInt() noexcept = default;

    /// sign-extends the given value, which must fit in 128 bits
    template <typename T>
    requires detail::cFitsFastInt128<T>
    constexpr FastInt( T v ) noexcept
    {
        const FastInt128 x = FastInt128( v );
        w[0] = std::uint64_t( x );
        w[1] = std::uint64_t( x >> 64 );
        const auto s = detail::signWord( w[1] );
        for ( int i = 2; i < numWords; ++i )
            w[i] = s;
    }

    /// sign-extends a narrower value of this family
    template <int mBits>
    requires ( mBits < nBits )
    constexpr FastInt( const FastInt<mBits> & v ) noexcept
    {
        constexpr int m = FastInt<mBits>::numWords;
        for ( int i = 0; i < m; ++i )
            w[i] = v.w[i];
        const auto s = detail::signWord( v.w[m - 1] );
        for ( int i = m; i < numWords; ++i )
            w[i] = s;
    }

    /// takes the lowest nBits bits of a wider value of this family, which must be enough to
    /// represent it exactly; as every product below widens, this is how a value with a proven
    /// bound returns to the narrow type where it belongs
    template <int mBits>
    requires ( mBits > nBits )
    constexpr explicit FastInt( const FastInt<mBits> & v ) noexcept
    {
        for ( int i = 0; i < numWords; ++i )
            w[i] = v.w[i];
        [[maybe_unused]] const auto s = detail::signWord( w[numWords - 1] );
        for ( int i = numWords; i < FastInt<mBits>::numWords; ++i )
            assert( v.w[i] == s ); // the value does not fit in nBits bits
    }

    /// -1, 0 or 1 if the value is negative, zero or positive respectively
    [[nodiscard]] constexpr int sign() const noexcept
    {
        if ( std::int64_t( w[numWords - 1] ) < 0 )
            return -1;
        std::uint64_t any = 0;
        for ( int i = 0; i < numWords; ++i )
            any |= w[i];
        return any != 0 ? 1 : 0;
    }

    constexpr FastInt & operator +=( const FastInt & b ) noexcept
    {
        std::uint64_t carry = 0;
        for ( int i = 0; i < numWords; ++i )
            w[i] = detail::addCarry64( w[i], b.w[i], carry );
        return *this;
    }

    constexpr FastInt & operator -=( const FastInt & b ) noexcept
    {
        std::uint64_t borrow = 0;
        for ( int i = 0; i < numWords; ++i )
            w[i] = detail::subBorrow64( w[i], b.w[i], borrow );
        return *this;
    }

    [[nodiscard]] constexpr FastInt operator -() const noexcept
    {
        FastInt res;
        res -= *this;
        return res;
    }

    [[nodiscard]] friend constexpr FastInt operator +( FastInt a, const FastInt & b ) noexcept { a += b; return a; }
    [[nodiscard]] friend constexpr FastInt operator -( FastInt a, const FastInt & b ) noexcept { a -= b; return a; }

    [[nodiscard]] friend constexpr bool operator ==( const FastInt & a, const FastInt & b ) noexcept { return a.w == b.w; }

    [[nodiscard]] friend constexpr std::strong_ordering operator <=>( const FastInt & a, const FastInt & b ) noexcept
    {
        if ( const auto c = std::int64_t( a.w[numWords - 1] ) <=> std::int64_t( b.w[numWords - 1] ); c != std::strong_ordering::equal )
            return c;
        for ( int i = numWords - 2; i >= 0; --i )
            if ( const auto c = a.w[i] <=> b.w[i]; c != std::strong_ordering::equal )
                return c;
        return std::strong_ordering::equal;
    }
};

/// the nearest double to the given value, with the same guarantees as toDouble( FastInt128 ) above
template <int nBits>
[[nodiscard]] MR_BIND_IGNORE inline double toDouble( const FastInt<nBits> & v ) noexcept
{
    return detail::doubleFromWords( v.w );
}

using FastInt256 = FastInt<256>;
using FastInt512 = FastInt<512>;
using FastInt1024 = FastInt<1024>;

/// every product below is exact, because the type of a product is as wide as the sum of
/// the widths of its arguments; a value with a proven bound is brought back to a narrow type
/// by the explicit narrowing constructor above

template <int nBits, int mBits>
[[nodiscard]] MR_BIND_IGNORE constexpr FastInt<nBits + mBits> operator *( const FastInt<nBits> & a, const FastInt<mBits> & b ) noexcept
{
    FastInt<nBits + mBits> res;
    res.w = detail::mulWords( a.w, b.w );
    return res;
}

template <int nBits, typename T>
requires detail::cFitsFastInt128<T>
[[nodiscard]] MR_BIND_IGNORE constexpr FastInt<nBits + detail::cMulBits<T>> operator *( const FastInt<nBits> & a, T b ) noexcept
{
    FastInt<nBits + detail::cMulBits<T>> res;
    res.w = detail::mulWords( a.w, detail::mulWordsOf( b ) );
    return res;
}

template <int nBits, typename T>
requires detail::cFitsFastInt128<T>
[[nodiscard]] MR_BIND_IGNORE constexpr FastInt<nBits + detail::cMulBits<T>> operator *( T a, const FastInt<nBits> & b ) noexcept
{
    return b * a;
}

/// a 128-bit integer, which product with another one is an exact 256-bit integer;
/// addition, subtraction and division stay 128-bit and can overflow just like FastInt128
class MR_BIND_IGNORE Int128Mul256
{
public:
    Int128Mul256() noexcept = default;

    template <typename T>
    requires detail::cFitsFastInt128<T>
    constexpr Int128Mul256( T v ) noexcept : v_( v ) { }

    [[nodiscard]] constexpr explicit operator FastInt128() const noexcept { return v_; }

    [[nodiscard]] friend constexpr FastInt128 operator +( Int128Mul256 a ) noexcept { return a.v_; }
    [[nodiscard]] friend constexpr FastInt128 operator -( Int128Mul256 a ) noexcept { return -a.v_; }

    [[nodiscard]] friend constexpr FastInt128 operator +( Int128Mul256 a, Int128Mul256 b ) noexcept { return a.v_ + b.v_; }
    [[nodiscard]] friend constexpr FastInt128 operator -( Int128Mul256 a, Int128Mul256 b ) noexcept { return a.v_ - b.v_; }
    [[nodiscard]] friend constexpr FastInt128 operator /( Int128Mul256 a, Int128Mul256 b ) noexcept { return a.v_ / b.v_; }

    [[nodiscard]] friend constexpr FastInt256 operator *( Int128Mul256 a, Int128Mul256 b ) noexcept
    {
        const FastUInt128 ua( a.v_ ), ub( b.v_ );
        FastInt256 res;
        res.w = detail::mulWords( std::array{ std::uint64_t( ua ), std::uint64_t( ua >> 64 ) },
                                  std::array{ std::uint64_t( ub ), std::uint64_t( ub >> 64 ) } );
        return res;
    }

    [[nodiscard]] friend constexpr bool operator ==( Int128Mul256 a, Int128Mul256 b ) noexcept { return a.v_ == b.v_; }
    [[nodiscard]] friend constexpr auto operator <=>( Int128Mul256 a, Int128Mul256 b ) noexcept { return a.v_ <=> b.v_; }

private:
    FastInt128 v_;
};

// no bindings for the same reason as for Vector3i128fast
#if !defined MR_PARSING_FOR_ANY_BINDINGS && !defined MR_COMPILING_ANY_BINDINGS
using Vector2i128mul = Vector2<Int128Mul256>;
using Vector3i128mul = Vector3<Int128Mul256>;
#endif

/// \}

} // namespace MR
