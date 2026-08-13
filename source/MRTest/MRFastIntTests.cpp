#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRHighPrecision.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

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

// detail::mulWords without any zero-word skip, kept as the oldest arm of
// DISABLED_FastIntMulWordsBench below; all arms must stay bit-identical
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

        // the product is exact in the twice wider type; the wider references are built from
        // the words and not converted from B, because GCC 12 gives false positive -Warray-bounds
        // on the widening conversions of boost integers
        EXPECT_EQ( toBoost<B2>( fa * fb ), toBoost<B2>( a ) * toBoost<B2>( b ) );
        EXPECT_EQ( toBoost<B2>( sqr( fa ) ), toBoost<B2>( a ) * toBoost<B2>( a ) );
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

TEST( MRMesh, FastInt192 )
{
    testFastIntWidth<192, Int256, Int512>( 444 ); // the narrowest width of the family
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
        const Int512 refA = toBoost<Int512>( a ), refB = toBoost<Int512>( b ), refE = toBoost<Int512>( e );

        // the width of a product is the sum of the widths of its arguments
        static_assert( std::is_same_v<decltype( fa * fb ), FastInt<448> > );
        static_assert( std::is_same_v<decltype( fa * std::int64_t( 1 ) ), FastInt<256> > );
        static_assert( std::is_same_v<decltype( fa * FastInt128( 1 ) ), FastInt<320> > );
        EXPECT_EQ( toBoost<Int512>( fa * fb ), refA * refB );
        EXPECT_EQ( toBoost<Int512>( fb * fa ), refA * refB );

        const auto c = std::int64_t( gen() );
        EXPECT_EQ( toBoost<Int512>( fa * c ), refA * Int512( c ) );
        EXPECT_EQ( toBoost<Int512>( c * fa ), refA * Int512( c ) );
        EXPECT_EQ( toBoost<Int512>( fa * toFastInt128( e ) ), refA * refE );

        // widening and narrowing back keeps the value
        EXPECT_TRUE( FastInt<192>( FastInt<448>( fa ) ) == fa );
        EXPECT_TRUE( FastInt<256>( FastInt<512>( fb ) ) == fb );
    }
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

// the order of the arguments must not change the product, which is what detail::mulWords relies
// on, and neither order may differ from the version without any zero-word skip
TEST( MRMesh, FastIntMulWordsOrder )
{
    std::mt19937_64 gen( 20260813 );
    for ( int i = 0; i < 20000; ++i )
    {
        const auto a = randomWords<4>( gen, 1 + i % 256 );
        const auto b = randomWords<4>( gen, 1 + ( 3 * i ) % 256 );
        const auto res = detail::mulWords( a, b );
        EXPECT_EQ( res, mulWordsRef( a, b ) );
        EXPECT_EQ( res, detail::mulWordsOrdered( a, b ) );
        EXPECT_EQ( res, detail::mulWordsOrdered( b, a ) );
        EXPECT_EQ( res, detail::mulWords( b, a ) );
        // and the same for arguments of different widths
        const Words<2> a2{ a[0], a[1] };
        const Words<1> b1{ b[0] };
        const auto res21 = detail::mulWords( a2, b1 );
        EXPECT_EQ( res21, mulWordsRef( a2, b1 ) );
        EXPECT_EQ( res21, detail::mulWords( b1, a2 ) );
    }
}

namespace
{

// nPairs argument pairs, of which only wordsA and wordsB low words are random and the rest are 0;
// zero word counts mean a random width in [1, n] and [1, m] respectively, per pair
template <std::size_t n, std::size_t m>
std::vector<std::pair<Words<n>, Words<m>>> mulWordsPairs(
    std::mt19937_64 & gen, std::size_t nPairs, std::size_t wordsA, std::size_t wordsB )
{
    std::vector<std::pair<Words<n>, Words<m>>> res;
    res.reserve( nPairs );
    for ( std::size_t i = 0; i < nPairs; ++i )
    {
        Words<n> a = {};
        Words<m> b = {};
        const std::size_t na = wordsA ? wordsA : 1 + gen() % n;
        const std::size_t nb = wordsB ? wordsB : 1 + gen() % m;
        for ( std::size_t k = 0; k < na; ++k )
            a[k] = gen();
        for ( std::size_t k = 0; k < nb; ++k )
            b[k] = gen();
        // keep both arguments non-negative, so that sign correction does not skew the timing
        a[na - 1] &= ~( std::uint64_t( 1 ) << 63 );
        b[nb - 1] &= ~( std::uint64_t( 1 ) << 63 );
        res.emplace_back( a, b );
    }
    return res;
}

// arm 0 = the version without any zero-word skip, 1 = master (skips the zero words of the first
// argument only, which is detail::mulWordsOrdered), 2 = the shipped detail::mulWords
template <int arm, std::size_t n, std::size_t m>
std::array<std::uint64_t, n + m> mulWordsArm( const Words<n> & a, const Words<m> & b ) noexcept
{
    if constexpr ( arm == 0 )
        return mulWordsRef( a, b );
    else if constexpr ( arm == 1 )
        return detail::mulWordsOrdered( a, b );
    else
        return detail::mulWords( a, b );
}

template <int arm, std::size_t n, std::size_t m>
double mulWordsMs( const std::vector<std::pair<Words<n>, Words<m>>> & pairs )
{
    std::uint64_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for ( const auto & [a, b] : pairs )
    {
        const auto r = mulWordsArm<arm>( a, b );
        sink ^= r[0] ^ r[n + m - 1];
    }
    const auto t1 = std::chrono::steady_clock::now();
    volatile std::uint64_t keep = sink; (void)keep;
    return std::chrono::duration<double, std::milli>( t1 - t0 ).count();
}

template <std::size_t n, std::size_t m>
void mulWordsBench( const char * name, const std::vector<std::pair<Words<n>, Words<m>>> & pairs )
{
    constexpr int warmup = 3, iters = 25;
    for ( int i = 0; i < warmup; ++i )
    {
        mulWordsMs<0>( pairs );
        mulWordsMs<1>( pairs );
        mulWordsMs<2>( pairs );
    }
    double orig = 1e300, master = 1e300, opt = 1e300;
    for ( int i = 0; i < iters; ++i ) // the minimum is the least noisy estimator here
    {
        orig = std::min( orig, mulWordsMs<0>( pairs ) );
        master = std::min( master, mulWordsMs<1>( pairs ) );
        opt = std::min( opt, mulWordsMs<2>( pairs ) );
    }
    std::printf( "[BENCH] mulWords %-22s pairs=%zu  noskip=%8.3f  master=%8.3f  new=%8.3f ms  vs master=%.2fx\n",
        name, pairs.size(), orig, master, opt, master / opt );
    std::fflush( stdout );
}

} // namespace

// opt-in A/B/C micro-benchmark for the zero-word skips in detail::mulWords: times the version
// without any skip and the master one (the first argument only) against the shipped
// detail::mulWords, which orders its arguments, over small-magnitude values (high words zero ->
// rows to skip), each argument small in turn, and full-width values (control, nothing to skip).
// Same opt-in idiom as DISABLED_PlanarTriangulationBench:
//   MRTest --gtest_also_run_disabled_tests --gtest_filter=*FastIntMulWordsBench*
TEST( MRMesh, DISABLED_FastIntMulWordsBench )
{
    constexpr std::size_t nPairs = 200000;
    std::mt19937_64 gen( 20240813 );
    mulWordsBench( "4x4 both 1 word", mulWordsPairs<4, 4>( gen, nPairs, 1, 1 ) );
    mulWordsBench( "4x4 second 1 word", mulWordsPairs<4, 4>( gen, nPairs, 4, 1 ) );
    mulWordsBench( "4x4 second 2 words", mulWordsPairs<4, 4>( gen, nPairs, 4, 2 ) );
    mulWordsBench( "4x4 second 3 words", mulWordsPairs<4, 4>( gen, nPairs, 4, 3 ) );
    mulWordsBench( "4x4 first 1 word", mulWordsPairs<4, 4>( gen, nPairs, 1, 4 ) );
    mulWordsBench( "4x4 random widths", mulWordsPairs<4, 4>( gen, nPairs, 0, 0 ) );
    mulWordsBench( "4x4 full (control)", mulWordsPairs<4, 4>( gen, nPairs, 4, 4 ) );
    mulWordsBench( "4x1 scalar", mulWordsPairs<4, 1>( gen, nPairs, 0, 1 ) );
    mulWordsBench( "2x2 both 1 word", mulWordsPairs<2, 2>( gen, nPairs, 1, 1 ) ); // Int128Mul256
    mulWordsBench( "2x2 second 1 word", mulWordsPairs<2, 2>( gen, nPairs, 2, 1 ) );
    mulWordsBench( "2x2 full (control)", mulWordsPairs<2, 2>( gen, nPairs, 2, 2 ) );
}

} // namespace MR
