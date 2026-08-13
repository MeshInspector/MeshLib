#include <MRMesh/MRFastInt.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <compare>
#include <cstdio>
#include <limits>
#include <ostream>
#include <random>
#include <string>
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
            std::snprintf( buf, sizeof( buf ), i == v.mag_.size() ? "%x" : "%08x", v.mag_[i - 1] );
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

} // namespace MR
