#include <MRMesh/MRLinesLoad.h>
#include <MRMesh/MRLinesSave.h>
#include <MRMesh/MRPolyline.h>
#include <gtest/gtest.h>
#include <cmath>
#include <sstream>

namespace MR
{

TEST( MRMesh, LinesDxfRoundTrip )
{
    const Contours3f contours{
        { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 2 } },
        { { 5, 0, 1 }, { 6, 0, 1 }, { 6, 1, 1 }, { 5, 0, 1 } } // closed
    };
    std::stringstream ss;
    ASSERT_TRUE( LinesSave::toDxf( Polyline3( contours ), ss ).has_value() );

    const auto loaded = LinesLoad::fromDxf( ss );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    EXPECT_EQ( loaded->contours(), contours );
}

TEST( MRMesh, LinesDxfEntities )
{
    // Windows line ends and padded group codes, as written by CAD programs
    const std::string dxf =
        "  0\r\nSECTION\r\n  2\r\nBLOCKS\r\n"
        "  0\r\nLINE\r\n 10\r\n100\r\n 20\r\n100\r\n 11\r\n200\r\n 21\r\n200\r\n"
        "  0\r\nENDSEC\r\n"
        "  0\r\nSECTION\r\n  2\r\nENTITIES\r\n"
        "  0\r\nLINE\r\n  8\r\n0\r\n 10\r\n1.0\r\n 20\r\n2.0\r\n 30\r\n3.0\r\n 11\r\n4.0\r\n 21\r\n5.0\r\n 31\r\n6.0\r\n"
        // closed square at the elevation 7
        "  0\r\nLWPOLYLINE\r\n 90\r\n4\r\n 70\r\n1\r\n 38\r\n7.0\r\n"
        " 10\r\n0\r\n 20\r\n0\r\n 10\r\n1\r\n 20\r\n0\r\n 10\r\n1\r\n 20\r\n1\r\n 10\r\n0\r\n 20\r\n1\r\n"
        // half-circle from (2,0) to (0,0) through (1,1)
        "  0\r\nLWPOLYLINE\r\n 90\r\n2\r\n 70\r\n0\r\n 10\r\n2\r\n 20\r\n0\r\n 42\r\n1.0\r\n 10\r\n0\r\n 20\r\n0\r\n"
        "  0\r\nCIRCLE\r\n 10\r\n10\r\n 20\r\n0\r\n 30\r\n0\r\n 40\r\n2.0\r\n"
        // quarter of a circle crossing zero angle
        "  0\r\nARC\r\n 10\r\n0\r\n 20\r\n0\r\n 30\r\n0\r\n 40\r\n1.0\r\n 50\r\n315\r\n 51\r\n45\r\n"
        // circle in the plane orthogonal to X axis
        "  0\r\nCIRCLE\r\n 10\r\n0\r\n 20\r\n0\r\n 30\r\n5\r\n 40\r\n1.0\r\n210\r\n1\r\n220\r\n0\r\n230\r\n0\r\n"
        "  0\r\nENDSEC\r\n  0\r\nEOF\r\n";

    std::istringstream in( dxf );
    const auto loaded = LinesLoad::fromDxf( in );
    ASSERT_TRUE( loaded.has_value() ) << loaded.error();
    const auto cs = loaded->contours();
    ASSERT_EQ( cs.size(), 6 );

    EXPECT_EQ( cs[0], ( Contour3f{ { 1, 2, 3 }, { 4, 5, 6 } } ) );
    EXPECT_EQ( cs[1], ( Contour3f{ { 0, 0, 7 }, { 1, 0, 7 }, { 1, 1, 7 }, { 0, 1, 7 }, { 0, 0, 7 } } ) );

    const float eps = 1e-5f;
    EXPECT_EQ( cs[2].front(), Vector3f( 2, 0, 0 ) );
    EXPECT_EQ( cs[2].back(), Vector3f( 0, 0, 0 ) );
    ASSERT_GT( cs[2].size(), 10 );
    for ( const auto& p : cs[2] )
    {
        EXPECT_NEAR( ( p - Vector3f( 1, 0, 0 ) ).length(), 1, eps );
        EXPECT_GE( p.y, -eps );
    }

    EXPECT_EQ( cs[3].front(), cs[3].back() );
    EXPECT_EQ( cs[3].size(), 73 );
    for ( const auto& p : cs[3] )
        EXPECT_NEAR( ( p - Vector3f( 10, 0, 0 ) ).length(), 2, eps );

    EXPECT_NEAR( ( cs[4].front() - Vector3f( std::sqrt( 0.5f ), -std::sqrt( 0.5f ), 0 ) ).length(), 0, eps );
    EXPECT_NEAR( ( cs[4].back() - Vector3f( std::sqrt( 0.5f ), std::sqrt( 0.5f ), 0 ) ).length(), 0, eps );
    for ( const auto& p : cs[4] )
        EXPECT_GT( p.x, 0.7f );

    // the center (0,0,5) in the object coordinate system with Z along world X is (5,0,0) in the world
    for ( const auto& p : cs[5] )
    {
        EXPECT_NEAR( p.x, 5, eps );
        EXPECT_NEAR( ( p - Vector3f( 5, 0, 0 ) ).length(), 1, eps );
    }
}

TEST( MRMesh, LinesDxfNoLines )
{
    std::istringstream in( "0\nSECTION\n2\nENTITIES\n0\nPOINT\n10\n1\n20\n2\n30\n3\n0\nENDSEC\n0\nEOF\n" );
    const auto loaded = LinesLoad::fromDxf( in );
    ASSERT_FALSE( loaded.has_value() );
    // so that the points loader gets its chance
    EXPECT_TRUE( loaded.error().starts_with( stringUnsupportedFileFormat() ) ) << loaded.error();
}

} //namespace MR
