#pragma once

#include "MRMeshFwd.h"
#include "MRFastInt128.h"
#include <MRPch/MRBindingMacros.h>
#include <array>
#include <cassert>
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

/// the exact product of two two's-complement values given by their 64-bit words, computed with
/// the inner loop over the mEff lowest words of b only; mEff is a template parameter and not an
/// argument, because a run-time trip count costs the unrolling this multiplication relies on
template <std::size_t mEff, std::size_t n, std::size_t m>
[[nodiscard]] constexpr std::array<std::uint64_t, n + m> mulWordsFixed(
    const std::array<std::uint64_t, n> & a, const std::array<std::uint64_t, m> & b ) noexcept
{
    static_assert( mEff >= 1 && mEff <= m );
    std::array<std::uint64_t, n + m> res = {};
    for ( std::size_t i = 0; i < n; ++i ) // schoolbook multiplication of unsigned values
    {
        if ( a[i] == 0 )
            continue; // this row adds nothing: 0 * b[j] leaves every res[i + j] unchanged, and
                      // res[i + mEff] is still 0 (never written before this row), so the skipped
                      // res[i + mEff] = carry (carry stays 0 here) would be a no-op. Small-magnitude
                      // values keep their high words at 0, so this is the common fast path; negative
                      // operands are sign-extended to all-ones top words and are not skipped.
        std::uint64_t carry = 0;
        for ( std::size_t j = 0; j < mEff; ++j )
        {
            // at most ( 2^64 - 1 )^2 + 2 * ( 2^64 - 1 ) < 2^128 here
            const FastUInt128 t = FastUInt128( a[i] ) * FastUInt128( b[j] ) + FastUInt128( res[i + j] ) + FastUInt128( carry );
            res[i + j] = std::uint64_t( t );
            carry = std::uint64_t( t >> 64 );
        }
        // never written before, since row i - 1 stopped at i - 1 + mEff; dropping the columns
        // j in [mEff, m), where b[j] = 0, is exact: the full loop would only have carried carry
        // into the still-zero res[i + mEff], where it fits without a carry-out, and then stored
        // zeros over zeros up to res[i + m]
        res[i + mEff] = carry;
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

/// the exact product of two two's-complement values given by their 64-bit words,
/// which always fits in the sum of their word counts; the only multiplication of this file
template <std::size_t n, std::size_t m> // std::size_t and not int, to be deducible from std::array
[[nodiscard]] constexpr std::array<std::uint64_t, n + m> mulWords(
    const std::array<std::uint64_t, n> & a, const std::array<std::uint64_t, m> & b ) noexcept
{
    if constexpr ( m > 1 )
    {
        // whether b fits in its lowest word, tested once per product and not in every inner loop
        // iteration, which was measured to cost more than it saves. Such a b is non-negative, as
        // its top word is 0, so its sign correction is skipped anyway; a negative b is
        // sign-extended to all-ones top words and never takes this path. Only this narrowest case
        // is singled out, because every extra width instantiates another copy of the whole
        // multiplication above.
        bool bOneWord = true;
        for ( std::size_t j = 1; j < m; ++j )
            if ( b[j] != 0 )
            {
                bOneWord = false; // the loop stops at the first significant word, which for a
                break;            // full-width b is the very first one it looks at
            }
        if ( bOneWord )
            return mulWordsFixed<1>( a, b );
    }
    return mulWordsFixed<m>( a, b );
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

} // namespace detail

/// signed integer of nBits bits, which must be a multiple of 64 and at least 192
/// (below that use FastInt128 with Int64Mul128 and Int128Mul256);
/// the product of two of them is exact, because it is twice as wide as the arguments;
/// as FastInt128 it lacks conversion in double, sqrt-function and stream input/output
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
