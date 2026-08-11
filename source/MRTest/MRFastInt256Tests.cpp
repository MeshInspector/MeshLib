#include <MRMesh/MRFastInt256.h>
#include <MRMesh/MRHighPrecision.h>
#include <gtest/gtest.h>
#include <random>

namespace MR
{

namespace
{

// the same value as a boost integer, which unlike FastInt256 can be printed and divided
Int256 toInt256( const FastInt256 & v )
{
    const bool neg = v.sign() < 0;
    auto w = v.w;
    if ( neg )
    {
        uint64_t borrow = 0;
        for ( int i = 0; i < 4; ++i )
            w[i] = detail::subBorrow64( 0, w[i], borrow );
    }
    Int256 res = 0;
    for ( int i = 3; i >= 0; --i )
    {
        res <<= 64;
        res |= Int256( w[i] );
    }
    return neg ? -res : res;
}

// two's complement 128-bit value given by its 64-bit words
FastInt128 toFastInt128( uint64_t lo, uint64_t hi )
{
    return FastInt128( ( FastUInt128( hi ) << 64 ) | FastUInt128( lo ) );
}

// the same value as a boost integer
Int256 toInt256( uint64_t lo, uint64_t hi )
{
    Int256 res = ( Int256( hi ) << 64 ) | Int256( lo );
    if ( int64_t( hi ) < 0 )
        res -= Int256( 1 ) << 128;
    return res;
}

// a random value with a random number of significant bits, to cover all magnitudes
struct Random128
{
    uint64_t lo = 0, hi = 0;

    Random128( std::mt19937_64 & gen, int maxBits )
    {
        lo = gen();
        hi = gen();
        const int bits = int( gen() % ( maxBits + 1 ) );
        if ( bits == 0 )
            lo = hi = 0;
        else if ( bits < 64 )
        {
            hi = 0;
            lo &= ~uint64_t( 0 ) >> ( 64 - bits );
        }
        else if ( bits < 128 )
            hi &= bits == 64 ? uint64_t( 0 ) : ~uint64_t( 0 ) >> ( 128 - bits );
        if ( gen() & 1 )
        {
            uint64_t borrow = 0;
            lo = detail::subBorrow64( 0, lo, borrow );
            hi = detail::subBorrow64( 0, hi, borrow );
        }
    }

    FastInt128 fast() const { return toFastInt128( lo, hi ); }
    Int256 ref() const { return toInt256( lo, hi ); }
};

constexpr FastInt128 cMin128( FastUInt128( 1 ) << 127 );
constexpr FastInt128 cMax128( ~( FastUInt128( 1 ) << 127 ) );

static_assert( mulExact( FastInt128( 3 ), FastInt128( -5 ) ) == -15 );
static_assert( mulExact( cMin128, cMin128 ).sign() > 0 );

} // anonymous namespace

TEST( MRMesh, FastInt256Basics )
{
    EXPECT_EQ( FastInt256().sign(), 0 );
    EXPECT_EQ( FastInt256( 0 ).sign(), 0 );
    EXPECT_EQ( FastInt256( 5 ).sign(), 1 );
    EXPECT_EQ( FastInt256( -5 ).sign(), -1 );
    EXPECT_TRUE( FastInt256( 0 ) == 0 );
    EXPECT_TRUE( FastInt256( -1 ) < 0 );
    EXPECT_TRUE( FastInt256( 1 ) > 0 );
    EXPECT_TRUE( FastInt256( -1 ) < FastInt256( 1 ) );

    // unsigned arguments are not sign-extended
    EXPECT_TRUE( FastInt256( ~uint64_t( 0 ) ) > 0 );
    EXPECT_EQ( toInt256( FastInt256( ~uint64_t( 0 ) ) ), ( Int256( 1 ) << 64 ) - 1 );

    EXPECT_EQ( toInt256( FastInt256( cMin128 ) ), -( Int256( 1 ) << 127 ) );
    EXPECT_EQ( toInt256( -FastInt256( cMin128 ) ), Int256( 1 ) << 127 );
    EXPECT_EQ( toInt256( FastInt256( 3 ) - FastInt256( 10 ) ), Int256( -7 ) );
    EXPECT_EQ( toInt256( FastInt256( -3 ) + FastInt256( 10 ) ), Int256( 7 ) );
}

TEST( MRMesh, FastInt256MulExact )
{
    EXPECT_EQ( toInt256( mulExact( 0, cMax128 ) ), Int256( 0 ) );
    EXPECT_EQ( toInt256( mulExact( cMin128, -1 ) ), Int256( 1 ) << 127 );
    EXPECT_EQ( toInt256( mulExact( cMin128, cMin128 ) ), Int256( 1 ) << 254 );
    EXPECT_EQ( toInt256( mulExact( cMin128, cMax128 ) ), -( ( Int256( 1 ) << 254 ) - ( Int256( 1 ) << 127 ) ) );
    EXPECT_EQ( toInt256( mulExact( cMax128, cMax128 ) ), ( ( Int256( 1 ) << 127 ) - 1 ) * ( ( Int256( 1 ) << 127 ) - 1 ) );

    std::mt19937_64 gen( 12345 );
    for ( int i = 0; i < 3000; ++i )
    {
        const Random128 a( gen, 128 ), b( gen, 128 );
        EXPECT_EQ( toInt256( mulExact( a.fast(), b.fast() ) ), a.ref() * b.ref() );
    }
}

TEST( MRMesh, FastInt256AddSubCompare )
{
    std::mt19937_64 gen( 54321 );
    for ( int i = 0; i < 3000; ++i )
    {
        // 120 bits at most, so that the sums below are far from the 256-bit limit
        const Random128 a0( gen, 120 ), a1( gen, 120 ), b0( gen, 120 ), b1( gen, 120 );
        const auto a = mulExact( a0.fast(), a1.fast() );
        const auto b = mulExact( b0.fast(), b1.fast() );
        const Int256 refA = a0.ref() * a1.ref(), refB = b0.ref() * b1.ref();

        EXPECT_EQ( toInt256( a ), refA );
        EXPECT_EQ( toInt256( a + b ), refA + refB );
        EXPECT_EQ( toInt256( a - b ), refA - refB );
        EXPECT_EQ( toInt256( -a ), -refA );
        EXPECT_EQ( a.sign(), refA.sign() );
        EXPECT_EQ( a == b, refA == refB );
        EXPECT_EQ( a < b, refA < refB );
        EXPECT_EQ( a > b, refA > refB );
        EXPECT_EQ( a <= b, refA <= refB );
    }
}

} // namespace MR
