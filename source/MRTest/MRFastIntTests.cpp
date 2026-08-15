#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstdio>
#include <limits>
#include <ostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace MR
{

namespace
{

/// an independent reference implementation to validate the FastInt family against:
/// an arbitrary-precision sign-magnitude integer over base-2^32 limbs with schoolbook
/// arithmetic. It is deliberately naive and shares no code with the classes under test,
/// so that a bug in them cannot hide in the reference as well. The 32-bit limbs keep every
/// intermediate value within std::uint64_t, so the reference needs no wide arithmetic itself.
class RefInt
{
public:
    RefInt() noexcept = default;

    RefInt( std::int64_t v )
    {
        if ( v == 0 )
            return;
        sign_ = v < 0 ? -1 : 1;
        // the magnitude via unsigned negation, which is correct for the smallest value as well
        const std::uint64_t m = v < 0 ? ~std::uint64_t( v ) + 1 : std::uint64_t( v );
        mag_ = { std::uint32_t( m ), std::uint32_t( m >> 32 ) };
        trim( mag_ );
    }

    /// the value with the given magnitude in 64-bit words, least significant first
    static RefInt fromMagnitude( const std::uint64_t * w, std::size_t nWords, bool negative )
    {
        RefInt res;
        res.mag_.reserve( 2 * nWords );
        for ( std::size_t i = 0; i < nWords; ++i )
        {
            res.mag_.push_back( std::uint32_t( w[i] ) );
            res.mag_.push_back( std::uint32_t( w[i] >> 32 ) );
        }
        trim( res.mag_ );
        res.sign_ = res.mag_.empty() ? 0 : ( negative ? -1 : 1 );
        return res;
    }

    [[nodiscard]] int sign() const { return sign_; }

    [[nodiscard]] RefInt operator-() const
    {
        RefInt res = *this;
        res.sign_ = -res.sign_;
        return res;
    }

    [[nodiscard]] friend RefInt operator+( const RefInt & a, const RefInt & b )
    {
        if ( a.sign_ == 0 )
            return b;
        if ( b.sign_ == 0 )
            return a;
        RefInt res;
        if ( a.sign_ == b.sign_ )
        {
            res.mag_ = addMag( a.mag_, b.mag_ );
            res.sign_ = a.sign_;
            return res;
        }
        const int c = cmpMag( a.mag_, b.mag_ ); // the signs differ, so the magnitudes subtract
        if ( c == 0 )
            return res;
        res.mag_ = c > 0 ? subMag( a.mag_, b.mag_ ) : subMag( b.mag_, a.mag_ );
        res.sign_ = c > 0 ? a.sign_ : b.sign_;
        return res;
    }

    [[nodiscard]] friend RefInt operator-( const RefInt & a, const RefInt & b ) { return a + -b; }

    [[nodiscard]] friend RefInt operator*( const RefInt & a, const RefInt & b )
    {
        RefInt res;
        if ( a.sign_ == 0 || b.sign_ == 0 )
            return res;
        res.mag_ = mulMag( a.mag_, b.mag_ );
        res.sign_ = a.sign_ * b.sign_;
        return res;
    }

    RefInt & operator+=( const RefInt & b ) { return *this = *this + b; }
    RefInt & operator-=( const RefInt & b ) { return *this = *this - b; }

    [[nodiscard]] RefInt operator<<( int bits ) const
    {
        RefInt res;
        if ( sign_ == 0 )
            return res;
        res.sign_ = sign_;
        res.mag_.assign( bits / 32, 0 );
        std::uint64_t carry = 0;
        for ( std::uint32_t d : mag_ )
        {
            const std::uint64_t cur = ( std::uint64_t( d ) << ( bits % 32 ) ) | carry;
            res.mag_.push_back( std::uint32_t( cur ) );
            carry = cur >> 32;
        }
        if ( carry != 0 )
            res.mag_.push_back( std::uint32_t( carry ) );
        return res;
    }

    friend bool operator==( const RefInt & a, const RefInt & b ) = default;

    [[nodiscard]] friend std::strong_ordering operator<=>( const RefInt & a, const RefInt & b )
    {
        if ( a.sign_ != b.sign_ )
            return a.sign_ <=> b.sign_;
        const int c = cmpMag( a.mag_, b.mag_ );
        return a.sign_ < 0 ? ( 0 <=> c ) : ( c <=> 0 ); // the larger magnitude is the smaller negative
    }

    /// in hexadecimal, since the reference has no division to print decimal digits with
    friend std::ostream & operator<<( std::ostream & s, const RefInt & v )
    {
        if ( v.sign_ == 0 )
            return s << "0x0";
        std::string str = v.sign_ < 0 ? "-0x" : "0x";
        for ( std::size_t i = v.mag_.size(); i > 0; --i )
        {
            char buf[9];
            // the format has to be a literal in each branch: MSVC rejects a ternary one (C4774)
            if ( i == v.mag_.size() )
                std::snprintf( buf, sizeof( buf ), "%x", v.mag_[i - 1] ); // the top limb, no leading zeros
            else
                std::snprintf( buf, sizeof( buf ), "%08x", v.mag_[i - 1] );
            str += buf;
        }
        return s << str;
    }

private:
    using Mag = std::vector<std::uint32_t>; // base 2^32, least significant limb first, no leading zeros

    static void trim( Mag & m )
    {
        while ( !m.empty() && m.back() == 0 )
            m.pop_back();
    }

    static int cmpMag( const Mag & a, const Mag & b )
    {
        if ( a.size() != b.size() )
            return a.size() < b.size() ? -1 : 1;
        for ( std::size_t i = a.size(); i > 0; --i )
            if ( a[i - 1] != b[i - 1] )
                return a[i - 1] < b[i - 1] ? -1 : 1;
        return 0;
    }

    static Mag addMag( const Mag & a, const Mag & b )
    {
        Mag res;
        std::uint64_t carry = 0;
        for ( std::size_t i = 0; i < std::max( a.size(), b.size() ) || carry != 0; ++i )
        {
            const std::uint64_t cur = carry
                + ( i < a.size() ? a[i] : 0 )
                + ( i < b.size() ? b[i] : 0 );
            res.push_back( std::uint32_t( cur ) );
            carry = cur >> 32;
        }
        return res;
    }

    static Mag subMag( const Mag & a, const Mag & b ) // requires a >= b
    {
        Mag res;
        std::int64_t borrow = 0;
        for ( std::size_t i = 0; i < a.size(); ++i )
        {
            std::int64_t cur = std::int64_t( a[i] ) - borrow - ( i < b.size() ? std::int64_t( b[i] ) : 0 );
            borrow = cur < 0 ? 1 : 0;
            if ( cur < 0 )
                cur += std::int64_t( 1 ) << 32;
            res.push_back( std::uint32_t( cur ) );
        }
        trim( res );
        return res;
    }

    static Mag mulMag( const Mag & a, const Mag & b )
    {
        Mag res( a.size() + b.size(), 0 );
        for ( std::size_t i = 0; i < a.size(); ++i )
        {
            std::uint64_t carry = 0;
            for ( std::size_t j = 0; j < b.size(); ++j )
            {
                // the largest term here is 2^64-1, so a 64-bit accumulator suffices
                const std::uint64_t cur = res[i + j] + std::uint64_t( a[i] ) * b[j] + carry;
                res[i + j] = std::uint32_t( cur );
                carry = cur >> 32;
            }
            for ( std::size_t j = b.size(); carry != 0; ++j )
            {
                const std::uint64_t cur = res[i + j] + carry;
                res[i + j] = std::uint32_t( cur );
                carry = cur >> 32;
            }
        }
        trim( res );
        return res;
    }

    int sign_ = 0; // -1, 0 or 1
    Mag mag_;      // the magnitude, empty if and only if the value is zero
};

template <std::size_t nWords> // std::size_t and not int, to be deducible from std::array
using Words = std::array<std::uint64_t, nWords>;

// negates the two's complement value in place, without the code under test
template <std::size_t nWords>
void negateWords( Words<nWords> & w )
{
    std::uint64_t carry = 1;
    for ( std::size_t i = 0; i < nWords; ++i )
    {
        w[i] = ~w[i] + carry;
        carry = carry != 0 && w[i] == 0 ? 1 : 0;
    }
}

// the same value as a reference integer, of any magnitude and printable
template <std::size_t nWords>
RefInt toRef( Words<nWords> w )
{
    const bool neg = std::int64_t( w[nWords - 1] ) < 0;
    if ( neg )
        negateWords( w ); // the magnitude, which the reference stores separately from the sign
    return RefInt::fromMagnitude( w.data(), nWords, neg );
}

template <int nBits>
RefInt toRef( const FastInt<nBits> & v )
{
    return toRef( v.w );
}

template <int nBits>
FastInt<nBits> toFastInt( const Words<nBits / 64> & w )
{
    FastInt<nBits> res;
    res.w = w;
    return res;
}

FastInt128 toFastInt128( const Words<2> & w )
{
    return FastInt128( ( FastUInt128( w[1] ) << 64 ) | FastUInt128( w[0] ) );
}

// a random value with a random number of significant bits, to cover all magnitudes
template <std::size_t nWords>
Words<nWords> randomWords( std::mt19937_64 & gen, int maxBits )
{
    Words<nWords> w = {};
    const int bits = int( gen() % ( maxBits + 1 ) );
    for ( std::size_t i = 0; i < nWords; ++i )
    {
        const int lowBit = 64 * int( i );
        if ( bits <= lowBit )
            break;
        w[i] = gen();
        if ( bits < lowBit + 64 )
            w[i] &= ~std::uint64_t( 0 ) >> ( lowBit + 64 - bits );
    }
    if ( gen() & 1 )
        negateWords( w );
    return w;
}

// the pre-optimization detail::mulWords (no zero-high-word skip), kept only as the reference
// arm of DISABLED_FastIntMulWordsBench below; both arms must stay bit-identical
template <std::size_t n, std::size_t m>
std::array<std::uint64_t, n + m> mulWordsRef(
    const std::array<std::uint64_t, n> & a, const std::array<std::uint64_t, m> & b ) noexcept
{
    std::array<std::uint64_t, n + m> res = {};
    for ( std::size_t i = 0; i < n; ++i )
    {
        std::uint64_t carry = 0;
        for ( std::size_t j = 0; j < m; ++j )
        {
            const FastUInt128 t = FastUInt128( a[i] ) * FastUInt128( b[j] ) + FastUInt128( res[i + j] ) + FastUInt128( carry );
            res[i + j] = std::uint64_t( t );
            carry = std::uint64_t( t >> 64 );
        }
        res[i + m] = carry;
    }
    if ( std::int64_t( a[n - 1] ) < 0 )
    {
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < m; ++i )
            res[n + i] = detail::subBorrow64( res[n + i], b[i], borrow );
    }
    if ( std::int64_t( b[m - 1] ) < 0 )
    {
        std::uint64_t borrow = 0;
        for ( std::size_t i = 0; i < n; ++i )
            res[m + i] = detail::subBorrow64( res[m + i], a[i], borrow );
    }
    return res;
}

constexpr FastInt128 cMin128( FastUInt128( 1 ) << 127 );
constexpr FastInt128 cMax128( ~( FastUInt128( 1 ) << 127 ) );

static_assert( Int128Mul256( 3 ) * Int128Mul256( -5 ) == -15 );
static_assert( ( Int128Mul256( cMin128 ) * Int128Mul256( cMin128 ) ).sign() > 0 );
static_assert( FastInt256( -3 ) * FastInt256( 5 ) == FastInt512( -15 ) );
static_assert( FastInt512( FastInt256( -7 ) ) == -7 );

} // anonymous namespace

// the reference itself, on the cases the tests below depend on
TEST( MRMesh, FastIntRefInt )
{
    EXPECT_EQ( RefInt( 0 ).sign(), 0 );
    EXPECT_EQ( RefInt( 5 ).sign(), 1 );
    EXPECT_EQ( RefInt( -5 ).sign(), -1 );
    EXPECT_EQ( RefInt( 7 ) - RefInt( 7 ), RefInt( 0 ) ); // exact cancellation is zero, not minus zero
    EXPECT_EQ( ( RefInt( 7 ) - RefInt( 7 ) ).sign(), 0 );
    EXPECT_EQ( RefInt( 3 ) - RefInt( 10 ), RefInt( -7 ) );
    EXPECT_EQ( RefInt( -3 ) * RefInt( -4 ), RefInt( 12 ) );
    EXPECT_EQ( RefInt( -3 ) * RefInt( 0 ), RefInt( 0 ) );
    EXPECT_TRUE( RefInt( -10 ) < RefInt( -9 ) ); // the larger magnitude is the smaller negative
    EXPECT_TRUE( RefInt( -1 ) < RefInt( 0 ) );
    EXPECT_TRUE( RefInt( 9 ) <= RefInt( 9 ) );

    // the smallest 64-bit value, whose magnitude does not fit in a positive one
    const auto cMin64 = std::numeric_limits<std::int64_t>::min();
    EXPECT_EQ( RefInt( cMin64 ), -( RefInt( 1 ) << 63 ) );
    EXPECT_EQ( RefInt( cMin64 ) * RefInt( -1 ), RefInt( 1 ) << 63 );

    // shifts across and within limbs, and the carry chain of the magnitude
    EXPECT_EQ( ( RefInt( 1 ) << 128 ) - ( RefInt( 1 ) << 127 ), RefInt( 1 ) << 127 );
    EXPECT_EQ( ( RefInt( 1 ) << 100 ) * ( RefInt( 1 ) << 156 ), RefInt( 1 ) << 256 );
    EXPECT_TRUE( ( RefInt( 1 ) << 256 ) > ( RefInt( 1 ) << 255 ) );

    const auto cAllOnes64 = ~std::uint64_t( 0 );
    const Words<2> maxU128{ cAllOnes64, cAllOnes64 }; // as a magnitude, not a negative value
    EXPECT_EQ( RefInt::fromMagnitude( maxU128.data(), 2, false ), ( RefInt( 1 ) << 128 ) - 1 );
    EXPECT_EQ( RefInt::fromMagnitude( maxU128.data(), 2, true ), 1 - ( RefInt( 1 ) << 128 ) );

    // the two's complement words of -1 are all ones, of any width
    const Words<3> minusOne{ cAllOnes64, cAllOnes64, cAllOnes64 };
    EXPECT_EQ( toRef( minusOne ), RefInt( -1 ) );
}

TEST( MRMesh, FastIntBasics )
{
    EXPECT_EQ( FastInt256::numWords, 4 );
    EXPECT_EQ( FastInt512::numWords, 8 );
    EXPECT_EQ( FastInt1024::numWords, 16 );

    EXPECT_EQ( FastInt1024().sign(), 0 );
    EXPECT_EQ( FastInt512( 0 ).sign(), 0 );
    EXPECT_EQ( FastInt256( 5 ).sign(), 1 );
    EXPECT_EQ( FastInt256( -5 ).sign(), -1 );
    EXPECT_TRUE( FastInt1024( -1 ) < 0 );
    EXPECT_TRUE( FastInt512( 1 ) > 0 );
    EXPECT_TRUE( FastInt256( -1 ) < FastInt256( 1 ) );

    // unsigned arguments are not sign-extended
    EXPECT_TRUE( FastInt256( ~std::uint64_t( 0 ) ) > 0 );
    EXPECT_EQ( toRef( FastInt256( ~std::uint64_t( 0 ) ) ), ( RefInt( 1 ) << 64 ) - 1 );

    EXPECT_EQ( toRef( FastInt256( cMin128 ) ), -( RefInt( 1 ) << 127 ) );
    EXPECT_EQ( toRef( -FastInt256( cMin128 ) ), RefInt( 1 ) << 127 );
    EXPECT_EQ( toRef( FastInt256( 3 ) - FastInt256( 10 ) ), RefInt( -7 ) );
    EXPECT_EQ( toRef( FastInt512( -3 ) + FastInt512( 10 ) ), RefInt( 7 ) );

    // widening from a narrower value of the family keeps the value
    EXPECT_EQ( toRef( FastInt1024( FastInt256( cMin128 ) ) ), -( RefInt( 1 ) << 127 ) );
    EXPECT_EQ( toRef( FastInt512( FastInt256( 12345 ) ) ), RefInt( 12345 ) );
}

TEST( MRMesh, FastIntInt128Mul256 )
{
    EXPECT_EQ( toRef( Int128Mul256( 0 ) * Int128Mul256( cMax128 ) ), RefInt( 0 ) );
    EXPECT_EQ( toRef( Int128Mul256( cMin128 ) * Int128Mul256( -1 ) ), RefInt( 1 ) << 127 );
    EXPECT_EQ( toRef( Int128Mul256( cMin128 ) * Int128Mul256( cMin128 ) ), RefInt( 1 ) << 254 );
    EXPECT_EQ( toRef( Int128Mul256( cMin128 ) * Int128Mul256( cMax128 ) ),
        -( ( RefInt( 1 ) << 254 ) - ( RefInt( 1 ) << 127 ) ) );
    EXPECT_EQ( toRef( Int128Mul256( cMax128 ) * Int128Mul256( cMax128 ) ),
        ( ( RefInt( 1 ) << 127 ) - 1 ) * ( ( RefInt( 1 ) << 127 ) - 1 ) );

    // the operations other than multiplication stay 128-bit
    EXPECT_TRUE( Int128Mul256( 7 ) + Int128Mul256( 3 ) == FastInt128( 10 ) );
    EXPECT_TRUE( Int128Mul256( 7 ) - Int128Mul256( 3 ) == FastInt128( 4 ) );
    EXPECT_TRUE( Int128Mul256( 7 ) / Int128Mul256( 3 ) == FastInt128( 2 ) );
    EXPECT_TRUE( -Int128Mul256( 7 ) == FastInt128( -7 ) );
    EXPECT_TRUE( Int128Mul256( 7 ) > Int128Mul256( 3 ) );

    std::mt19937_64 gen( 12345 );
    for ( int i = 0; i < 3000; ++i )
    {
        const auto a = randomWords<2>( gen, 128 ), b = randomWords<2>( gen, 128 );
        const RefInt refA = toRef( a ), refB = toRef( b );
        EXPECT_EQ( toRef( Int128Mul256( toFastInt128( a ) ) * Int128Mul256( toFastInt128( b ) ) ), refA * refB );
        EXPECT_EQ( toRef( sqr( Int128Mul256( toFastInt128( a ) ) ) ), refA * refA );
    }
}

// sums, comparisons and the exact product of the given width against the reference
template <int nBits>
void testFastIntWidth( unsigned seed )
{
    constexpr int n = nBits / 64;
    std::mt19937_64 gen( seed );
    for ( int i = 0; i < 1000; ++i )
    {
        // one bit short of the full width, so that the sums below cannot overflow
        const auto a = randomWords<n>( gen, nBits - 1 ), b = randomWords<n>( gen, nBits - 1 );
        const auto fa = toFastInt<nBits>( a ), fb = toFastInt<nBits>( b );
        const RefInt refA = toRef( a ), refB = toRef( b );

        EXPECT_EQ( toRef( fa ), refA );
        EXPECT_EQ( toRef( fa + fb ), refA + refB );
        EXPECT_EQ( toRef( fa - fb ), refA - refB );
        EXPECT_EQ( toRef( -fa ), -refA );
        EXPECT_EQ( fa.sign(), refA.sign() );
        EXPECT_EQ( fa == fb, refA == refB );
        EXPECT_EQ( fa < fb, refA < refB );
        EXPECT_EQ( fa > fb, refA > refB );
        EXPECT_EQ( fa <= fb, refA <= refB );

        // the product is exact in the twice wider type, and the reference grows with the value
        EXPECT_EQ( toRef( fa * fb ), refA * refB );
        EXPECT_EQ( toRef( sqr( fa ) ), refA * refA );
    }
}

TEST( MRMesh, FastInt256 )
{
    testFastIntWidth<256>( 111 );
}

TEST( MRMesh, FastInt512 )
{
    testFastIntWidth<512>( 222 );
}

TEST( MRMesh, FastInt1024 )
{
    testFastIntWidth<1024>( 333 );
}

TEST( MRMesh, FastInt192 )
{
    testFastIntWidth<192>( 444 ); // the narrowest width of the family
}

TEST( MRMesh, FastIntMixedWidths )
{
    std::mt19937_64 gen( 777 );
    for ( int i = 0; i < 1000; ++i )
    {
        const Words<3> a = randomWords<3>( gen, 192 );
        const Words<4> b = randomWords<4>( gen, 256 );
        const Words<2> e = randomWords<2>( gen, 128 );
        const auto fa = toFastInt<192>( a );
        const auto fb = toFastInt<256>( b );
        const RefInt refA = toRef( a ), refB = toRef( b ), refE = toRef( e );

        // the width of a product is the sum of the widths of its arguments
        static_assert( std::is_same_v<decltype( fa * fb ), FastInt<448> > );
        static_assert( std::is_same_v<decltype( fa * std::int64_t( 1 ) ), FastInt<256> > );
        static_assert( std::is_same_v<decltype( fa * FastInt128( 1 ) ), FastInt<320> > );
        EXPECT_EQ( toRef( fa * fb ), refA * refB );
        EXPECT_EQ( toRef( fb * fa ), refA * refB );

        const auto c = std::int64_t( gen() );
        EXPECT_EQ( toRef( fa * c ), refA * RefInt( c ) );
        EXPECT_EQ( toRef( c * fa ), refA * RefInt( c ) );
        EXPECT_EQ( toRef( fa * toFastInt128( e ) ), refA * refE );

        // widening and narrowing back keeps the value
        EXPECT_TRUE( FastInt<192>( FastInt<448>( fa ) ) == fa );
        EXPECT_TRUE( FastInt<256>( FastInt<512>( fb ) ) == fb );
    }
}

namespace
{

// the value of a finite double as a reference integer, exact for every integer double
RefInt toRef( double d )
{
    if ( std::abs( d ) < std::ldexp( 1.0, 53 ) )
        return RefInt( std::int64_t( d ) ); // every integer of this magnitude fits a std::int64_t
    int e = 0;
    const double m = std::frexp( d, &e ); // d == m * 2^e with |m| in [0.5, 1)
    const auto mi = std::int64_t( std::ldexp( m, 53 ) ); // exact: 53 bits is the whole mantissa
    return RefInt( mi ) << ( e - 53 ); // and e >= 54 here, so the shift is never negative
}

// |a - b|, as the reference has no absolute value of its own
RefInt refDist( const RefInt & a, const RefInt & b )
{
    const RefInt d = a - b;
    return d.sign() < 0 ? -d : d;
}

// toDouble of the given words is the nearest double to their value: within the documented
// relative error of 2^-53, and with no neighbouring double strictly closer
template <std::size_t nWords>
void expectNearestDouble( const Words<nWords> & w, double d )
{
    const RefInt ref = toRef( w );
    ASSERT_EQ( d < 0, ref.sign() < 0 ) << ref;
    ASSERT_EQ( d == 0, ref.sign() == 0 ) << ref;
    if ( ref.sign() == 0 )
        return;
    ASSERT_TRUE( std::isfinite( d ) ) << ref; // no width here reaches DBL_MAX, see FastIntToDoubleExtremes
    if ( std::abs( d ) < std::ldexp( 1.0, 53 ) )
    {
        EXPECT_EQ( toRef( d ), ref ) << "exact below 2^53"; // the mantissa holds the value whole
        return;
    }
    const RefInt refD = toRef( d );
    const RefInt err = refDist( ref, refD );
    const RefInt absRef = ref.sign() < 0 ? -ref : ref;

    // correct rounding is at most half an ulp off, which is 2^-53 of the value at worst
    EXPECT_LE( err << 53, absRef ) << "value " << ref << " became " << refD;

    // and no other double is closer, which the bound alone does not imply
    EXPECT_GE( refDist( ref, toRef( std::nextafter( d, -INFINITY ) ) ), err ) << ref;
    EXPECT_GE( refDist( ref, toRef( std::nextafter( d, INFINITY ) ) ), err ) << ref;
}

// toDouble of the given width against the reference, over all magnitudes
template <int nBits>
void testToDoubleWidth( unsigned seed )
{
    constexpr int n = nBits / 64;
    std::mt19937_64 gen( seed );
    for ( int i = 0; i < 1000; ++i )
    {
        const auto w = randomWords<n>( gen, nBits - 1 ); // one bit short, so that -v does not overflow
        const auto v = toFastInt<nBits>( w );
        const double d = toDouble( v );
        expectNearestDouble( w, d );
        EXPECT_EQ( toDouble( -v ), -d ); // the conversion is symmetric, being sign-magnitude
        EXPECT_EQ( d > 0, v.sign() > 0 );
        EXPECT_EQ( d < 0, v.sign() < 0 );
    }
}

} // anonymous namespace

// every value of magnitude below 2^53 converts exactly, at every width of the family
TEST( MRMesh, FastIntToDoubleExact )
{
    std::mt19937_64 gen( 555 );
    const auto test = [&]( std::int64_t v )
    {
        const auto d = double( v ); // exact for |v| < 2^53, and the conversion must agree with it
        EXPECT_EQ( toDouble( FastInt128( v ) ), d );
        EXPECT_EQ( toDouble( FastInt<192>( v ) ), d );
        EXPECT_EQ( toDouble( FastInt256( v ) ), d );
        EXPECT_EQ( toDouble( FastInt1024( v ) ), d );
    };
    for ( std::int64_t v : { std::int64_t( 0 ), std::int64_t( 1 ), std::int64_t( -1 ), std::int64_t( 2 ),
        std::int64_t( 1 ) << 52, -( std::int64_t( 1 ) << 52 ), ( std::int64_t( 1 ) << 53 ) - 1 } )
        test( v );
    for ( int i = 0; i < 1000; ++i )
    {
        const auto v = std::int64_t( gen() % ( std::uint64_t( 1 ) << 53 ) );
        test( v );
        test( -v );
    }
    EXPECT_EQ( toDouble( FastInt256( 0 ) ), 0 );
    EXPECT_FALSE( std::signbit( toDouble( FastInt256( 0 ) ) ) ); // zero has no sign here
}

// the first magnitudes that do not fit the mantissa, where the rounding starts
TEST( MRMesh, FastIntToDoubleRounding )
{
    const auto pow53 = std::int64_t( 1 ) << 53;
    EXPECT_EQ( toDouble( FastInt256( pow53 ) ), std::ldexp( 1.0, 53 ) );
    EXPECT_EQ( toDouble( FastInt256( pow53 + 1 ) ), std::ldexp( 1.0, 53 ) );      // ties to even, down
    EXPECT_EQ( toDouble( FastInt256( pow53 + 2 ) ), std::ldexp( 1.0, 53 ) + 2 );  // exact again
    EXPECT_EQ( toDouble( FastInt256( pow53 + 3 ) ), std::ldexp( 1.0, 53 ) + 4 );  // ties to even, up
    EXPECT_EQ( toDouble( FastInt256( -pow53 - 1 ) ), -std::ldexp( 1.0, 53 ) );
    EXPECT_EQ( toDouble( FastInt256( -pow53 - 3 ) ), -std::ldexp( 1.0, 53 ) - 4 );

    // a tie broken by a single bit far below the mantissa, which only the sticky bit can carry:
    // 2^64 + 2^11 is a tie of the doubles around it, and 2^64 + 2^11 + 1 is just above it
    const FastInt256 v = FastInt256( FastInt128( FastUInt128( 1 ) << 64 ) ) + FastInt256( std::int64_t( 1 ) << 11 );
    EXPECT_EQ( toDouble( v ), std::ldexp( 1.0, 64 ) ); // ties to even, down
    EXPECT_EQ( toDouble( v + FastInt256( 1 ) ), std::ldexp( 1.0, 64 ) + 4096 ); // above the tie, up
}

TEST( MRMesh, FastIntToDouble128 )
{
    std::mt19937_64 gen( 666 );
    for ( int i = 0; i < 1000; ++i )
    {
        const auto w = randomWords<2>( gen, 127 ); // one bit short, so that -v does not overflow
        const auto v = toFastInt128( w );
        const double d = toDouble( v );
        expectNearestDouble( w, d );
        EXPECT_EQ( toDouble( FastInt128( -v ) ), -d );
        // the wider types hold the same value and must convert to the very same double
        EXPECT_EQ( toDouble( FastInt256( v ) ), d );
        EXPECT_EQ( toDouble( FastInt1024( v ) ), d );
        // toDouble( FastInt128 ) defers to the built-in conversion where there is one, so this
        // is what keeps its MSVC implementation checked - and equal to the built-in - everywhere
        EXPECT_EQ( detail::doubleFromWords( w ), d );
    }
    EXPECT_EQ( toDouble( cMax128 ), std::ldexp( 1.0, 127 ) ); // rounds up to the power of two
    EXPECT_EQ( toDouble( cMin128 ), -std::ldexp( 1.0, 127 ) );
}

TEST( MRMesh, FastIntToDouble256 )
{
    testToDoubleWidth<256>( 111 );
}

TEST( MRMesh, FastIntToDouble512 )
{
    testToDoubleWidth<512>( 222 );
}

TEST( MRMesh, FastIntToDouble192 )
{
    testToDoubleWidth<192>( 444 ); // the narrowest width of the family
}

// the widest alias still fits a double, and past DBL_MAX the conversion saturates; that is
// reachable, because every product widens (FastInt1024 * FastInt1024 is a FastInt<2048>)
TEST( MRMesh, FastIntToDoubleExtremes )
{
    FastInt1024 max1024; // the largest value of the width: all ones but the sign bit, 2^1023 - 1
    max1024.w.fill( ~std::uint64_t( 0 ) );
    max1024.w.back() = ~std::uint64_t( 0 ) >> 1;
    const FastInt1024 min1024 = -max1024 - FastInt1024( 1 ); // and the smallest, -2^1023
    // both are below DBL_MAX = ( 2 - 2^-52 ) * 2^1023, and 2^1023 - 1 rounds up to the power of two
    EXPECT_EQ( toDouble( max1024 ), std::ldexp( 1.0, 1023 ) );
    EXPECT_EQ( toDouble( min1024 ), -std::ldexp( 1.0, 1023 ) );

    // a power of two just below DBL_MAX converts exactly, and 2^1024 no longer fits
    FastInt<2048> pow1023;
    pow1023.w[15] = std::uint64_t( 1 ) << 63; // positive here: the sign bit of this width is w[31]
    EXPECT_EQ( toDouble( pow1023 ), std::ldexp( 1.0, 1023 ) );
    FastInt<2048> pow1024;
    pow1024.w[16] = 1;
    EXPECT_EQ( toDouble( pow1024 ), INFINITY );
    EXPECT_EQ( toDouble( -pow1024 ), -INFINITY );

    // and so does the square of the widest alias, which is what reaches here in practice
    EXPECT_EQ( toDouble( max1024 * max1024 ), INFINITY );
    EXPECT_EQ( toDouble( max1024 * min1024 ), -INFINITY );

    testToDoubleWidth<1024>( 333 ); // the random values are one bit short of the width and stay finite
}

TEST( MRMesh, FastIntVector )
{
    std::mt19937_64 gen( 54321 );
    for ( int i = 0; i < 1000; ++i )
    {
        Words<2> a[3], b[3];
        for ( int j = 0; j < 3; ++j )
        {
            a[j] = randomWords<2>( gen, 100 );
            b[j] = randomWords<2>( gen, 100 );
        }
        const Vector3i128mul u{ toFastInt128( a[0] ), toFastInt128( a[1] ), toFastInt128( a[2] ) };
        const Vector3i128mul v{ toFastInt128( b[0] ), toFastInt128( b[1] ), toFastInt128( b[2] ) };

        // dot deduces FastInt256 as its result type, so the dot product is exact
        RefInt ref = 0;
        for ( int j = 0; j < 3; ++j )
            ref += toRef( a[j] ) * toRef( b[j] );
        EXPECT_EQ( toRef( dot( u, v ) ), ref );
        EXPECT_EQ( toRef( u.lengthSq() ), toRef( dot( u, u ) ) );
    }
}

// opt-in A/B micro-benchmark for the zero-high-word skip in detail::mulWords: times the
// pre-optimization reference (mulWordsRef) against the shipped detail::mulWords over many random
// 256-bit-wide products, for small-magnitude values (high words zero -> fast path) and full-width
// values (control, nothing to skip). Same opt-in idiom as DISABLED_PlanarTriangulationBench:
//   MRTest --gtest_also_run_disabled_tests --gtest_filter=*FastIntMulWordsBench*
TEST( MRMesh, DISABLED_FastIntMulWordsBench )
{
    constexpr std::size_t nPairs = 200000;
    constexpr int warmup = 3, iters = 25;
    std::mt19937_64 gen( 20240813 );

    // low-word-only, non-negative 4-word inputs: high 3 words are 0 -> 3 of 4 rows are skippable
    std::vector<std::pair<Words<4>, Words<4>>> small;
    small.reserve( nPairs );
    for ( std::size_t i = 0; i < nPairs; ++i )
    {
        Words<4> a = {}, b = {};
        a[0] = gen() >> 1; // clear the top bit so the value stays non-negative (high words 0)
        b[0] = gen() >> 1;
        small.emplace_back( a, b );
    }

    // full-width random 4-word inputs: every word set -> no row is skippable (control)
    std::vector<std::pair<Words<4>, Words<4>>> full;
    full.reserve( nPairs );
    for ( std::size_t i = 0; i < nPairs; ++i )
    {
        Words<4> a, b;
        for ( std::size_t k = 0; k < 4; ++k ) { a[k] = gen(); b[k] = gen(); }
        a[3] &= ~( std::uint64_t( 1 ) << 63 ); // keep non-negative so sign correction doesn't skew timing
        b[3] &= ~( std::uint64_t( 1 ) << 63 );
        full.emplace_back( a, b );
    }

    const auto timeMs = []( const std::vector<std::pair<Words<4>, Words<4>>> & pairs, bool optimized )
    {
        std::uint64_t sink = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for ( const auto & [a, b] : pairs )
        {
            const auto r = optimized ? detail::mulWords( a, b ) : mulWordsRef( a, b );
            sink ^= r[0] ^ r[7];
        }
        const auto t1 = std::chrono::steady_clock::now();
        volatile std::uint64_t keep = sink; (void)keep;
        return std::chrono::duration<double, std::milli>( t1 - t0 ).count();
    };

    const auto runAB = [&]( const char * name, const std::vector<std::pair<Words<4>, Words<4>>> & pairs )
    {
        for ( int i = 0; i < warmup; ++i ) { timeMs( pairs, false ); timeMs( pairs, true ); }
        double ref = 1e300, opt = 1e300;
        for ( int i = 0; i < iters; ++i )
        {
            ref = std::min( ref, timeMs( pairs, false ) );
            opt = std::min( opt, timeMs( pairs, true ) );
        }
        std::printf( "[BENCH] mulWords %-12s pairs=%zu  ref=%8.3f  opt=%8.3f ms  speedup=%.2fx\n",
            name, pairs.size(), ref, opt, ref / opt );
        std::fflush( stdout );
    };

    runAB( "small(hi=0)", small );
    runAB( "full-width", full );
}

} // namespace MR
