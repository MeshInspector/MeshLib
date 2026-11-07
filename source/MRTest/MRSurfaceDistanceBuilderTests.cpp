#include <MRMesh/MRSurfaceDistanceBuilder.h>
#include <MRMesh/MRSurfaceDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

namespace MR
{

TEST(MRMesh, SurfaceDistance)
{
    float vc = 0;
    EXPECT_FALSE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0, 1, 0 }, 1, vc ) );

    EXPECT_FALSE( getFieldAtC( Vector3f{ 2, 1, 0 }, Vector3f{ 3, 3, 0 }, 1, vc ) );

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.5f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.1f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 0.9f, 1, 0 }, 0, vc ) );
    EXPECT_NEAR( vc, 1, 1e-5f );
    vc = 0;

    EXPECT_TRUE( getFieldAtC( Vector3f{ 1, 0, 0 }, Vector3f{ 1, 0.5f, 0 }, 1 / std::sqrt(2.0f), vc ) );
    EXPECT_NEAR( vc, 1.5f / std::sqrt(2.0f), 1e-5f );
    vc = 0;
}

// planar n x n grid with unit step; inner vertices shifted by up to 0.3 if jitter is set, except for the row straightRow
static Mesh makePlanarGrid( int n, bool jitter, int straightRow = -1 )
{
    VertCoords points;
    for ( int y = 0; y < n; ++y )
        for ( int x = 0; x < n; ++x )
        {
            Vector3f p( float( x ), float( y ), 0 );
            if ( jitter && x > 0 && y > 0 && x + 1 < n && y + 1 < n && y != straightRow )
                p += 0.3f * Vector3f( std::sin( 12.9898f * x + 78.233f * y ), std::cos( 39.3468f * x + 11.135f * y ), 0 );
            points.push_back( p );
        }
    Triangulation t;
    for ( int y = 0; y + 1 < n; ++y )
        for ( int x = 0; x + 1 < n; ++x )
        {
            const VertId v( y * n + x );
            t.push_back( { v, v + 1, v + n + 1 } );
            t.push_back( { v, v + n + 1, v + n } );
        }
    return Mesh::fromTriangles( std::move( points ), t );
}

TEST( MRMesh, SurfaceDistancePointSource )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true );
    const VertId seed( n * n / 2 );
    VertBitSet seeds( mesh.topology.vertSize() );
    seeds.set( seed );
    const auto dist = computeSurfaceDistances( mesh, seeds );
    float maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
        maxErr = std::max( maxErr, std::abs( dist[v] - ( mesh.points[v] - mesh.points[seed] ).length() ) );
    EXPECT_LT( maxErr, 2e-3f );
}

TEST( MRMesh, SurfaceDistanceLineSource )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true, n / 2 );
    VertBitSet seeds( mesh.topology.vertSize() );
    for ( int x = 0; x < n; ++x )
        seeds.set( VertId( n / 2 * n + x ) );
    const auto dist = computeSurfaceDistances( mesh, seeds );
    float maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
        maxErr = std::max( maxErr, std::abs( dist[v] - std::abs( mesh.points[v].y - float( n / 2 ) ) ) );
    EXPECT_LT( maxErr, 2e-3f );
}

TEST( MRMesh, SurfaceDistanceTwoSources )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true );
    const VertId s0( n / 2 * n + 10 ), s1( n / 2 * n + 30 );
    VertBitSet seeds( mesh.topology.vertSize() );
    seeds.set( s0 );
    seeds.set( s1 );
    const auto dist = computeSurfaceDistances( mesh, seeds );
    // where the fronts meet, the distances come along the edges, so check the mean error
    float sumErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
        sumErr += std::abs( dist[v] - std::min( ( mesh.points[v] - mesh.points[s0] ).length(), ( mesh.points[v] - mesh.points[s1] ).length() ) );
    EXPECT_LT( sumErr / ( n * n ), 2e-3f );
}

} //namespace MR
