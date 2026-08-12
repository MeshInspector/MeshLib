#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRHighPrecision.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>
#include <random>

namespace MR
{

namespace
{

template <std::size_t nWords> // std::size_t and not int, to be deducible from std::array
using Words = std::array<std::uint64_t, nWords>;

// negates the two's complement value in place
template <std::size_t nWords>
void negateWords( Words<nWords> & w )
{
    std::uint64_t borrow = 0;
    for ( std::size_t i = 0; i < nWords; ++i )
        w[i] = detail::subBorrow64( 0, w[i], borrow );
}

// the same value as a boost integer, which unlike the classes here can be printed and divided
template <typename B, std::size_t nWords>
B toBoost( Words<nWords> w )
{
    const bool neg = std::int64_t( w[nWords - 1] ) < 0;
    if ( neg )
        negateWords( w ); // the magnitude, to never need 2^(64*nWords) in B
    B res = 0;
    for ( std::size_t i = nWords; i > 0; --i )
    {
        res <<= 64;
        res |= B( w[i - 1] );
    }
    return neg ? -res : res;
}

template <typename B, int nBits>
B toBoost( const FastInt<nBits> & v )
{
    return toBoost<B>( v.w );
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

constexpr FastInt128 cMin128( FastUInt128( 1 ) << 127 );
constexpr FastInt128 cMax128( ~( FastUInt128( 1 ) << 127 ) );

static_assert( Int128Mul256( 3 ) * Int128Mul256( -5 ) == -15 );
static_assert( ( Int128Mul256( cMin128 ) * Int128Mul256( cMin128 ) ).sign() > 0 );
static_assert( FastInt256( -3 ) * FastInt256( 5 ) == FastInt512( -15 ) );
static_assert( FastInt512( FastInt256( -7 ) ) == -7 );

} // anonymous namespace

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
    EXPECT_EQ( toBoost<Int256>( FastInt256( ~std::uint64_t( 0 ) ) ), ( Int256( 1 ) << 64 ) - 1 );

    EXPECT_EQ( toBoost<Int256>( FastInt256( cMin128 ) ), -( Int256( 1 ) << 127 ) );
    EXPECT_EQ( toBoost<Int256>( -FastInt256( cMin128 ) ), Int256( 1 ) << 127 );
    EXPECT_EQ( toBoost<Int256>( FastInt256( 3 ) - FastInt256( 10 ) ), Int256( -7 ) );
    EXPECT_EQ( toBoost<Int512>( FastInt512( -3 ) + FastInt512( 10 ) ), Int512( 7 ) );

    // widening from a narrower value of the family keeps the value
    EXPECT_EQ( toBoost<Int1024>( FastInt1024( FastInt256( cMin128 ) ) ), -( Int1024( 1 ) << 127 ) );
    EXPECT_EQ( toBoost<Int512>( FastInt512( FastInt256( 12345 ) ) ), Int512( 12345 ) );
}

TEST( MRMesh, FastIntInt128Mul256 )
{
    EXPECT_EQ( toBoost<Int256>( Int128Mul256( 0 ) * Int128Mul256( cMax128 ) ), Int256( 0 ) );
    EXPECT_EQ( toBoost<Int256>( Int128Mul256( cMin128 ) * Int128Mul256( -1 ) ), Int256( 1 ) << 127 );
    EXPECT_EQ( toBoost<Int256>( Int128Mul256( cMin128 ) * Int128Mul256( cMin128 ) ), Int256( 1 ) << 254 );
    EXPECT_EQ( toBoost<Int256>( Int128Mul256( cMin128 ) * Int128Mul256( cMax128 ) ),
        -( ( Int256( 1 ) << 254 ) - ( Int256( 1 ) << 127 ) ) );
    EXPECT_EQ( toBoost<Int256>( Int128Mul256( cMax128 ) * Int128Mul256( cMax128 ) ),
        ( ( Int256( 1 ) << 127 ) - 1 ) * ( ( Int256( 1 ) << 127 ) - 1 ) );

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
        const Int256 refA = toBoost<Int256>( a ), refB = toBoost<Int256>( b );
        EXPECT_EQ( toBoost<Int256>( Int128Mul256( toFastInt128( a ) ) * Int128Mul256( toFastInt128( b ) ) ), refA * refB );
        EXPECT_EQ( toBoost<Int256>( sqr( Int128Mul256( toFastInt128( a ) ) ) ), refA * refA );
    }
}

// sums, comparisons and the exact product of the given width against boost
template <int nBits, typename B, typename B2>
void testFastIntWidth( unsigned seed )
{
    constexpr int n = nBits / 64;
    std::mt19937_64 gen( seed );
    for ( int i = 0; i < 1000; ++i )
    {
        // one bit short of the full width, so that the sums below cannot overflow
        const auto a = randomWords<n>( gen, nBits - 1 ), b = randomWords<n>( gen, nBits - 1 );
        const auto fa = toFastInt<nBits>( a ), fb = toFastInt<nBits>( b );
        const B refA = toBoost<B>( a ), refB = toBoost<B>( b );

        EXPECT_EQ( toBoost<B>( fa ), refA );
        EXPECT_EQ( toBoost<B>( fa + fb ), refA + refB );
        EXPECT_EQ( toBoost<B>( fa - fb ), refA - refB );
        EXPECT_EQ( toBoost<B>( -fa ), -refA );
        EXPECT_EQ( fa.sign(), refA.sign() );
        EXPECT_EQ( fa == fb, refA == refB );
        EXPECT_EQ( fa < fb, refA < refB );
        EXPECT_EQ( fa > fb, refA > refB );
        EXPECT_EQ( fa <= fb, refA <= refB );

        // the product is exact in the twice wider type
        EXPECT_EQ( toBoost<B2>( fa * fb ), B2( refA ) * B2( refB ) );
        EXPECT_EQ( toBoost<B2>( sqr( fa ) ), B2( refA ) * B2( refA ) );
    }
}

TEST( MRMesh, FastInt256 )
{
    testFastIntWidth<256, Int256, Int512>( 111 );
}

TEST( MRMesh, FastInt512 )
{
    testFastIntWidth<512, Int512, Int1024>( 222 );
}

TEST( MRMesh, FastInt1024 )
{
    using Int2048 = boost::multiprecision::number<boost::multiprecision::cpp_int_backend<2048, 2048,
        boost::multiprecision::signed_magnitude, boost::multiprecision::unchecked, void>>;
    testFastIntWidth<1024, Int1024, Int2048>( 333 );
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
        Int256 ref = 0;
        for ( int j = 0; j < 3; ++j )
            ref += toBoost<Int256>( a[j] ) * toBoost<Int256>( b[j] );
        EXPECT_EQ( toBoost<Int256>( dot( u, v ) ), ref );
        EXPECT_EQ( toBoost<Int256>( u.lengthSq() ), toBoost<Int256>( dot( u, u ) ) );
    }
}

} // namespace MR
