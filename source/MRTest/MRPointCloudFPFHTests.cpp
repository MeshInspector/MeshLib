#include <MRMesh/MRPointCloudFPFH.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMeshToPointCloud.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRMatrix3.h>
#include <MRMesh/MRConstants.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

namespace MR
{

namespace
{

PointCloud makePlaneCloud()
{
    PointCloud cloud;
    for ( int x = 0; x < 10; ++x )
        for ( int y = 0; y < 10; ++y )
            cloud.addPoint( Vector3f( float( x ), float( y ), 0 ), Vector3f::plusZ() );
    return cloud;
}

} // anonymous namespace

TEST( MRMesh, FPFHPlane )
{
    auto cloud = makePlaneCloud();
    auto res = computeFPFH( cloud, 2.5f );
    ASSERT_TRUE( res.has_value() );
    // all angles between equal normals and the lines in their plane are zero, which is the middle bin of every histogram
    for ( auto v : cloud.validPoints )
        for ( int i = 0; i < FPFHDescriptorSize; ++i )
            EXPECT_NEAR( (*res)[v][i], i % FPFHBins == FPFHBins / 2 ? 1.0f : 0.0f, 1e-6f );

    cloud.normals.clear();
    EXPECT_FALSE( computeFPFH( cloud, 2.5f ).has_value() );
}

TEST( MRMesh, FPFHInvariance )
{
    auto cloud = meshToPointCloud( makeTorus( 1.0f, 0.3f, 64, 32 ) );
    constexpr float radius = 0.25f;
    auto res0 = computeFPFH( cloud, radius );
    ASSERT_TRUE( res0.has_value() );

    // rotation, translation and uniform scaling together with the radius
    constexpr float scale = 3;
    auto moved = cloud;
    moved.transform( AffineXf3f( scale * Matrix3f::rotation( Vector3f( 1, 2, 3 ).normalized(), 0.7f ), Vector3f( 5, -7, 11 ) ) );
    for ( auto v : moved.validPoints )
        moved.normals[v] = moved.normals[v].normalized();
    auto res1 = computeFPFH( moved, scale * radius );
    ASSERT_TRUE( res1.has_value() );

    float maxDiff = 0;
    for ( auto v : cloud.validPoints )
    {
        float diff = 0;
        for ( int i = 0; i < FPFHDescriptorSize; ++i )
            diff += std::abs( (*res0)[v][i] - (*res1)[v][i] );
        maxDiff = std::max( maxDiff, diff );
    }
    EXPECT_LT( maxDiff, 0.05f );
}

TEST( MRMesh, FPFHUniquePoints )
{
    // 100 equal descriptors and a different one
    FPFHDescriptors descriptors( 101 );
    for ( auto & d : descriptors )
        d[FPFHBins / 2] = d[FPFHBins + FPFHBins / 2] = d[2 * FPFHBins + FPFHBins / 2] = 1;
    descriptors.back() = {};
    descriptors.back()[0] = descriptors.back()[FPFHBins] = descriptors.back()[2 * FPFHBins] = 1;

    VertBitSet points( descriptors.size() );
    points.set();
    auto unique = findUniqueFPFHPoints( descriptors, points, 2 );
    EXPECT_EQ( unique.count(), 1 );
    EXPECT_TRUE( unique.test( VertId( 100 ) ) );

    // only given points are considered
    points.reset( VertId( 100 ) );
    EXPECT_EQ( findUniqueFPFHPoints( descriptors, points, 2 ).count(), 0 );

    // all points of a plane are alike
    auto cloud = makePlaneCloud();
    auto res = findUniqueFPFHPoints( cloud, 2.5f, 1 );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( res->count(), 0 );

    cloud.normals.clear();
    EXPECT_FALSE( findUniqueFPFHPoints( cloud, 2.5f, 1 ).has_value() );
}

TEST( MRMesh, FPFHMatches )
{
    auto oneHot = []( int i )
    {
        FPFHDescriptor d{};
        d[i] = 1;
        return d;
    };
    FPFHDescriptors src, tgt;
    for ( int i = 0; i < 5; ++i )
        src.push_back( oneHot( i ) );
    for ( int i = 4; i >= 0; --i )
        tgt.push_back( oneHot( i ) );
    // its closest target point is the copy of descriptor 1, which is closer to source point 1
    auto extra = oneHot( 1 );
    extra[1] = 0.6f;
    src.push_back( extra );

    VertBitSet srcPoints( src.size() ), tgtPoints( tgt.size() );
    srcPoints.set();
    tgtPoints.set();
    auto all = findFPFHMatches( src, srcPoints, tgt, tgtPoints, false );
    ASSERT_TRUE( all.has_value() );
    ASSERT_EQ( all->size(), 6 );
    for ( int i = 0; i < 5; ++i )
        EXPECT_EQ( (*all)[i], VertPair( VertId( i ), VertId( 4 - i ) ) );
    EXPECT_EQ( (*all)[5], VertPair( VertId( 5 ), VertId( 3 ) ) );

    auto mutual = findFPFHMatches( src, srcPoints, tgt, tgtPoints, true );
    ASSERT_TRUE( mutual.has_value() );
    EXPECT_EQ( mutual->size(), 5 );
    EXPECT_EQ( std::count( mutual->begin(), mutual->end(), VertPair( VertId( 5 ), VertId( 3 ) ) ), 0 );

    // without source point 1, the extra point becomes the mutual match of its target
    srcPoints.reset( VertId( 1 ) );
    mutual = findFPFHMatches( src, srcPoints, tgt, tgtPoints, true );
    ASSERT_TRUE( mutual.has_value() );
    EXPECT_EQ( mutual->size(), 5 );
    EXPECT_EQ( std::count( mutual->begin(), mutual->end(), VertPair( VertId( 5 ), VertId( 3 ) ) ), 1 );

    EXPECT_TRUE( findFPFHMatches( src, srcPoints, tgt, VertBitSet( tgt.size() ) )->empty() );
}

} //namespace MR
