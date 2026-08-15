#include "MRFastInt.h"
#include <bit>
#include <cassert>

namespace MR
{

namespace detail
{

namespace
{

/// the largest exponent of a normal double, and so the largest step exp2i below can take
constexpr int cMaxDoubleExp = 1023;

/// 2^e as a double, for an exponent within [-1022, cMaxDoubleExp], that is of a normal double;
/// it is assembled as a bit pattern, because std::ldexp and std::exp2 are library calls, and one
/// of them costs about as much as everything else doubleFromWords below does
[[nodiscard]] inline double exp2i( int e ) noexcept
{
    assert( e >= -1022 && e <= cMaxDoubleExp );
    return std::bit_cast<double>( std::uint64_t( e + cMaxDoubleExp ) << 52 );
}

} // anonymous namespace

double doubleFromWords( const std::uint64_t * w, int n ) noexcept
{
    assert( n >= 1 );
    const bool neg = std::int64_t( w[n - 1] ) < 0;

    int lo = 0; // the lowest non-zero word
    while ( lo < n && w[lo] == 0 )
        ++lo;
    if ( lo == n )
        return 0; // and not -0 for a negative zero, which two's complement has no representation of

    // the words of the magnitude, computed on demand since w is read-only: negating a two's
    // complement value leaves every word below the lowest non-zero one at zero, negates that one,
    // and complements the ones above it. The smallest value negates into 2^(64*n-1), which is not
    // representable as a signed value here, but is as an unsigned magnitude
    const auto mag = [w, neg, lo]( int i ) noexcept -> std::uint64_t
    {
        return neg ? ( i <= lo ? ~w[i] + 1 : ~w[i] ) : w[i];
    };

    int k = n; // one past the highest non-zero word of the magnitude
    while ( mag( k - 1 ) == 0 )
        --k; // stops at lo + 1 at the latest, where the magnitude is non-zero as well

    // the top 64 significant bits, with the highest one in bit 63: the word below k contributes
    // the bits shifted in from the right, and the shift by s cannot lose anything, because the
    // top s bits of mag( k - 1 ) are zero by the definition of s
    const std::uint64_t hi = mag( k - 1 );
    const int s = std::countl_zero( hi );
    std::uint64_t top = hi << s;
    if ( s > 0 && k >= 2 )
        top |= mag( k - 2 ) >> ( 64 - s );

    // everything below those 64 bits, as a single sticky bit in the lowest one. That keeps the
    // conversion of top correctly rounded for the whole value: only 53 of its bits reach the
    // mantissa, so bit 0 sits well below the rounding position, and a non-zero tail there is
    // exactly what tells a tie (round to even) from a value strictly above it (round up)
    bool tail = k >= 2 && ( mag( k - 2 ) << s ) != 0; // the bits of mag( k - 2 ) that top did not take, all of it if s is 0
    for ( int i = 0; i + 2 < k && !tail; ++i )
        tail = mag( i ) != 0; // the words entirely below top
    if ( tail )
        top |= 1;

    // std::uint64_t -> double is correctly rounded, and the scaling by a power of two is exact
    // unless it overflows, which yields the infinity of the right sign as intended. A value past
    // DBL_MAX takes more than one step, its exponent alone exceeding what a double holds; every
    // factor below is at least one, so an intermediate infinity is one the exact value reaches too
    double res = double( top );
    int e = 64 * ( k - 1 ) - s; // at least -63, since k >= 1 and s <= 63, so nothing underflows
    for ( ; e > cMaxDoubleExp; e -= cMaxDoubleExp )
        res *= exp2i( cMaxDoubleExp );
    res *= exp2i( e );
    return neg ? -res : res;
}

} // namespace detail

} // namespace MR
