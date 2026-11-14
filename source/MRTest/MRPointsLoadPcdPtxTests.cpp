#include <MRMesh/MRPointsLoad.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRColor.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

namespace MR
{

namespace
{

template <typename T>
void append( std::string& s, T v )
{
    s.append( ( const char* )&v, sizeof( v ) );
}

} // anonymous namespace

TEST( MRMesh, PcdAscii )
{
    std::istringstream in(
        "# .PCD v0.7 - Point Cloud Data file format\r\n"
        "VERSION 0.7\r\n"
        "FIELDS intensity x y z rgb normal_x normal_y normal_z\r\n"
        "SIZE 4 4 4 4 4 4 4 4\r\n"
        "TYPE F F F F U F F F\r\n"
        "COUNT 1 1 1 1 1 1 1 1\r\n"
        "WIDTH 3\r\n"
        "HEIGHT 1\r\n"
        "VIEWPOINT 0 0 0 1 0 0 0\r\n"
        "POINTS 3\r\n"
        "DATA ascii\r\n"
        "0.5 1 2 3 16711680 0 0 1\r\n"
        "0.5 nan nan nan 0 0 0 1\r\n"
        "0.5 -4.5 5.25 6 255 0 1 0\r\n" );
    VertColors colors;
    const auto cloud = PointsLoad::fromPcd( in, { .colors = &colors } );
    ASSERT_TRUE( cloud.has_value() ) << cloud.error();
    ASSERT_EQ( cloud->points.size(), 3 );
    EXPECT_EQ( cloud->validPoints.count(), 2 );
    EXPECT_FALSE( cloud->validPoints.test( 1_v ) );
    EXPECT_EQ( cloud->points[0_v], Vector3f( 1, 2, 3 ) );
    EXPECT_EQ( cloud->points[2_v], Vector3f( -4.5f, 5.25f, 6 ) );
    ASSERT_EQ( cloud->normals.size(), 3 );
    EXPECT_EQ( cloud->normals[0_v], Vector3f( 0, 0, 1 ) );
    EXPECT_EQ( cloud->normals[2_v], Vector3f( 0, 1, 0 ) );
    ASSERT_EQ( colors.size(), 3 );
    EXPECT_EQ( colors[0_v], Color( 255, 0, 0 ) );
    EXPECT_EQ( colors[2_v], Color( 0, 0, 255 ) );
}

TEST( MRMesh, PcdBinary )
{
    // double coordinates and the color packed in a float, as many programs write it
    std::string s =
        "VERSION .7\nFIELDS x y z rgb\nSIZE 8 8 8 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        "WIDTH 2\nHEIGHT 2\nPOINTS 4\nDATA binary\n";
    for ( int i = 0; i < 4; ++i )
    {
        append( s, i == 1 ? std::numeric_limits<double>::quiet_NaN() : 1000000.0 + i );
        append( s, 2000000.0 );
        append( s, 0.25 * i );
        append( s, std::uint32_t( 0x00102030 + i ) );
    }
    std::istringstream in( s );
    VertColors colors;
    AffineXf3f xf;
    const auto cloud = PointsLoad::fromPcd( in, { .colors = &colors, .outXf = &xf } );
    ASSERT_TRUE( cloud.has_value() ) << cloud.error();
    ASSERT_EQ( cloud->points.size(), 4 );
    EXPECT_EQ( cloud->validPoints.count(), 3 );
    EXPECT_FALSE( cloud->validPoints.test( 1_v ) );
    // large coordinates go into the transformation
    EXPECT_EQ( xf.b, Vector3f( 1000000, 2000000, 0 ) );
    EXPECT_EQ( cloud->points[0_v], Vector3f() );
    EXPECT_EQ( cloud->points[3_v], Vector3f( 3, 0, 0.75f ) );
    EXPECT_EQ( colors[3_v], Color( 0x10, 0x20, 0x33 ) );

    // without the last point
    std::istringstream truncated( s.substr( 0, s.size() - 28 ) );
    EXPECT_FALSE( PointsLoad::fromPcd( truncated ).has_value() );
}

TEST( MRMesh, PcdBinaryCompressed )
{
    constexpr int n = 10;
    // the data are stored as arrays of fields: all x, then all y, then all z
    std::string raw;
    for ( int i = 0; i < n; ++i )
        append( raw, float( i ) );
    for ( int i = 0; i < n; ++i )
        append( raw, float( 10 * i ) );
    for ( int i = 0; i < n; ++i )
        append( raw, 5.0f );

    // LZF stream: literal runs for x and y, then 4 literal bytes of the first z and the copy of them for the other z
    std::string packed;
    for ( size_t pos = 0; pos < 80; pos += 32 )
    {
        const auto len = std::min<size_t>( 32, 80 - pos );
        packed.push_back( char( len - 1 ) );
        packed.append( raw, pos, len );
    }
    packed.push_back( char( 3 ) );
    packed.append( raw, 80, 4 );
    packed.push_back( '\xE0' ); // long copy with the high bits of distance equal to 0
    packed.push_back( char( 36 - 2 - 7 ) ); // 36 bytes
    packed.push_back( char( 4 - 1 ) ); // from the distance 4

    std::string s = "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 10\nHEIGHT 1\nPOINTS 10\nDATA binary_compressed\n";
    append( s, std::uint32_t( packed.size() ) );
    append( s, std::uint32_t( raw.size() ) );
    s += packed;

    std::istringstream in( s );
    const auto cloud = PointsLoad::fromPcd( in );
    ASSERT_TRUE( cloud.has_value() ) << cloud.error();
    ASSERT_EQ( cloud->points.size(), n );
    EXPECT_EQ( cloud->validPoints.count(), n );
    for ( int i = 0; i < n; ++i )
        EXPECT_EQ( cloud->points[VertId( i )], Vector3f( float( i ), float( 10 * i ), 5 ) );

    // corrupted stream
    s[s.size() - 1] = char( 100 );
    std::istringstream bad( s );
    EXPECT_FALSE( PointsLoad::fromPcd( bad ).has_value() );
}

TEST( MRMesh, Ptx )
{
    std::istringstream in(
        // the first scan is not moved
        "2\n2\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n"
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n"
        "1 2 3 0.5 255 0 0\n"
        "0 0 0 0.5 0 0 0\n"
        "4 5 6 0.25 0 255 0\n"
        "7 8 9 0.75 0 0 255\n"
        // the second scan is rotated by 90 degrees around Z and shifted
        "1\n2\n"
        "10 20 30\n0 1 0\n-1 0 0\n0 0 1\n"
        "0 1 0 0\n-1 0 0 0\n0 0 1 0\n10 20 30 1\n"
        "1 0 0 0.5 10 20 30\n"
        "0 2 0 0.5 40 50 60\n" );
    VertColors colors;
    const auto cloud = PointsLoad::fromPtx( in, { .colors = &colors } );
    ASSERT_TRUE( cloud.has_value() ) << cloud.error();
    ASSERT_EQ( cloud->points.size(), 6 );
    EXPECT_EQ( cloud->validPoints.count(), 5 );
    EXPECT_FALSE( cloud->validPoints.test( 1_v ) );
    EXPECT_EQ( cloud->points[0_v], Vector3f( 1, 2, 3 ) );
    EXPECT_EQ( cloud->points[3_v], Vector3f( 7, 8, 9 ) );
    // x axis of the second scan is Y of the common coordinates, its y axis is -X
    EXPECT_EQ( cloud->points[4_v], Vector3f( 10, 21, 30 ) );
    EXPECT_EQ( cloud->points[5_v], Vector3f( 8, 20, 30 ) );
    ASSERT_EQ( colors.size(), 6 );
    EXPECT_EQ( colors[0_v], Color( 255, 0, 0 ) );
    EXPECT_EQ( colors[5_v], Color( 40, 50, 60 ) );

    std::istringstream bad( "2\n2\n0 0 0\n" );
    EXPECT_FALSE( PointsLoad::fromPtx( bad ).has_value() );
}

} //namespace MR
