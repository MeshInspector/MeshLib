#include <MRIOExtras/MRLas.h>
#ifndef MRIOEXTRAS_NO_LAS
#include <MRMesh/MRColor.h>
#include <MRMesh/MRPointCloud.h>
#include <gtest/gtest.h>

#include <sstream>

namespace MR
{

namespace
{

PointCloud makeCloud( int n, VertColors& colors )
{
    PointCloud cloud;
    for ( int i = 0; i < n; ++i )
    {
        cloud.points.push_back( Vector3f( 1000.f + 0.01f * ( i % 1000 ), -50.f + 0.001f * i, float( i % 17 ) ) );
        cloud.normals.push_back( Vector3f( 0, float( i % 2 ), float( 1 - i % 2 ) ) );
        colors.push_back( Color( i % 256, 2 * i % 256, 255 - i % 256 ) );
    }
    cloud.validPoints.resize( cloud.points.size(), true );
    return cloud;
}

void checkRoundTrip( bool compress, int n )
{
    VertColors colors;
    auto cloud = makeCloud( n, colors );
    cloud.validPoints.reset( 1_v );

    std::stringstream ss;
    const SaveSettings settings{ .colors = &colors };
    ASSERT_TRUE( compress ? PointsSave::toLaz( cloud, ss, settings ) : PointsSave::toLas( cloud, ss, settings ) );

    VertColors loadedColors;
    auto loaded = PointsLoad::fromLas( ss, { .colors = &loadedColors } );
    ASSERT_TRUE( loaded ) << loaded.error();
    ASSERT_EQ( loaded->points.size(), size_t( n - 1 ) );
    ASSERT_EQ( loaded->normals.size(), size_t( n - 1 ) );
    ASSERT_EQ( loadedColors.size(), size_t( n - 1 ) );
    VertId lv( 0 );
    for ( auto v : cloud.validPoints )
    {
        EXPECT_NEAR( ( loaded->points[lv] - cloud.points[v] ).length(), 0.f, 1e-4f );
        EXPECT_EQ( loaded->normals[lv], cloud.normals[v] );
        EXPECT_EQ( loadedColors[lv], colors[v] );
        ++lv;
    }
}

} // anonymous namespace

TEST( MRIOExtras, LasPoints )
{
    // several write blocks of 50000 points each
    checkRoundTrip( false, 120000 );
}

TEST( MRIOExtras, LazPoints )
{
    // several compressed chunks of 50000 points each
    checkRoundTrip( true, 120000 );
}

TEST( MRIOExtras, LasPointsOnly )
{
    VertColors colors;
    auto cloud = makeCloud( 100, colors );
    cloud.normals.clear();

    for ( bool compress : { false, true } )
    {
        std::stringstream ss;
        ASSERT_TRUE( compress ? PointsSave::toLaz( cloud, ss ) : PointsSave::toLas( cloud, ss ) );
        auto loaded = PointsLoad::fromLas( ss );
        ASSERT_TRUE( loaded ) << loaded.error();
        ASSERT_EQ( loaded->points.size(), 100 );
        EXPECT_TRUE( loaded->normals.empty() );
        for ( auto v : cloud.validPoints )
            EXPECT_NEAR( ( loaded->points[v] - cloud.points[v] ).length(), 0.f, 1e-4f );
    }
}

} // namespace MR
#endif
