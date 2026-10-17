#include <MRMesh/MRRound.h>
#include <gtest/gtest.h>
#include <cmath>

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
            check( std::nextafter( x, 1e300 ) );
            check( std::nextafter( x, -1e300 ) );
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
            check( std::nextafter( x, 1e30f ) );
            check( std::nextafter( x, -1e30f ) );
        }
    }
    for ( float x : { 0.49999997f, -0.49999997f, 1e-40f, -1e-40f, 0.0f, -0.0f, 4194303.5f, -4194303.5f, 8388607.5f, -8388607.5f, 2147483520.0f, -2147483648.0f } )
        check( x );
}

} //namespace MR
