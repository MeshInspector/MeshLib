#include <MRMesh/MRRound.h>
#include <MRMesh/MRNextUpDown.h>
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

namespace MR
{

TEST( MRMesh, RoundToIntDouble )
{
    auto check = []( double x )
    {
        EXPECT_EQ( roundToInt( x ), (int)std::round( x ) ) << x;
    };
    for ( int i = -1000; i <= 1000; ++i )
    {
        for ( double x : { i + 0.5, double( i ) } )
        {
            check( x );
            check( nextUp( x ) );
            check( nextDown( x ) );
        }
    }
    for ( double x : { 0.49999999999999994, -0.49999999999999994, 1e-320, -1e-320, 0.0, -0.0, 1073741823.5, -1073741823.5, 2147483646.5, -2147483647.5 } )
        check( x );
}

TEST( MRMesh, RoundToIntFloat )
{
    auto check = []( float x )
    {
        EXPECT_EQ( roundToInt( x ), (int)std::round( x ) ) << x;
    };
    for ( int i = -1000; i <= 1000; ++i )
    {
        for ( float x : { i + 0.5f, float( i ) } )
        {
            check( x );
            check( nextUp( x ) );
            check( nextDown( x ) );
        }
    }
    for ( float x : { 0.49999997f, -0.49999997f, 1e-40f, -1e-40f, 0.0f, -0.0f, 4194303.5f, -4194303.5f, 8388607.5f, -8388607.5f, 2147483520.0f, -2147483648.0f } )
        check( x );
}

template <typename T, typename U>
static void checkNextUpDown()
{
    auto check = []( T x )
    {
        const T up = nextUp( x ), down = nextDown( x );
        const T refUp = std::nextafter( x, std::numeric_limits<T>::max() ), refDown = std::nextafter( x, std::numeric_limits<T>::lowest() );
        EXPECT_EQ( std::memcmp( &up, &refUp, sizeof( T ) ), 0 ) << x;
        EXPECT_EQ( std::memcmp( &down, &refDown, sizeof( T ) ), 0 ) << x;
    };
    using L = std::numeric_limits<T>;
    for ( T x : { T( 0 ), -T( 0 ), L::denorm_min(), -L::denorm_min(), L::min(), -L::min(), L::max(), L::lowest(), L::infinity(), -L::infinity(), T( 1 ), T( -1 ) } )
        check( x );
    std::mt19937_64 rnd( 1 );
    for ( int i = 0; i < 100000; ++i )
    {
        const auto u = U( rnd() );
        T x;
        std::memcpy( &x, &u, sizeof( T ) );
        if ( !std::isnan( x ) )
            check( x );
    }
}

TEST( MRMesh, NextUpDown )
{
    checkNextUpDown<float, std::uint32_t>();
    checkNextUpDown<double, std::uint64_t>();
}

} //namespace MR
