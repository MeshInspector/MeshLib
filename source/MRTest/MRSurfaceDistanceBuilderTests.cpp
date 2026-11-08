#include <MRMesh/MRSurfaceDistanceBuilder.h>
#include <MRMesh/MRSurfaceDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRegionBoundary.h>
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
    float sumErr = 0, maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const float err = std::abs( dist[v] - std::min( ( mesh.points[v] - mesh.points[s0] ).length(), ( mesh.points[v] - mesh.points[s1] ).length() ) );
        sumErr += err;
        maxErr = std::max( maxErr, err );
    }
    EXPECT_LT( sumErr / ( n * n ), 2e-4f );
    EXPECT_LT( maxErr, 2e-2f );
}

TEST( MRMesh, SurfaceDistanceTwoSegments )
{
    const int n = 41;
    const int row = n / 2;
    const int ends[2][2] = { { 5, 15 }, { 25, 35 } };
    const auto mesh = makePlanarGrid( n, true, row );
    VertBitSet seeds( mesh.topology.vertSize() );
    for ( const auto & e : ends )
        for ( int x = e[0]; x <= e[1]; ++x )
            seeds.set( VertId( row * n + x ) );
    const auto dist = computeSurfaceDistances( mesh, seeds );
    float sumErr = 0, maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const auto & p = mesh.points[v];
        float segDist[2];
        for ( int i = 0; i < 2; ++i )
            segDist[i] = std::hypot( std::max( { float( ends[i][0] ) - p.x, p.x - float( ends[i][1] ), 0.f } ), p.y - float( row ) );
        const float err = std::abs( dist[v] - std::min( segDist[0], segDist[1] ) );
        sumErr += err;
        maxErr = std::max( maxErr, err );
    }
    EXPECT_LT( sumErr / ( n * n ), 5e-3f );
    EXPECT_LT( maxErr, 5e-2f );
}

TEST( MRMesh, SurfaceDistanceEdgesAndPoints )
{
    const int n = 41;
    const int row = n / 2;
    const auto mesh = makePlanarGrid( n, true, row );
    VertBitSet lineVerts( mesh.topology.vertSize() );
    for ( int x = 5; x <= 15; ++x )
        lineVerts.set( VertId( row * n + x ) );
    const auto lineEdges = getInnerEdges( mesh.topology, lineVerts );
    // start edges between all start vertices give the same distances as these start vertices
    EXPECT_TRUE( computeSurfaceDistances( mesh, lineEdges, VertBitSet{} ) == computeSurfaceDistances( mesh, lineVerts ) );

    // start points one edge away from the segment: beyond its end and beside its middle
    const VertId pts[2] = { VertId( row * n + 16 ), VertId( ( row + 1 ) * n + 10 ) };
    VertBitSet startPoints( mesh.topology.vertSize() );
    for ( auto v : pts )
        startPoints.set( v );
    const auto dist = computeSurfaceDistances( mesh, lineEdges, startPoints );
    float sumErr = 0, maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const auto & p = mesh.points[v];
        float d[3] = {
            std::hypot( std::max( { 5.f - p.x, p.x - 15.f, 0.f } ), p.y - float( row ) ),
            ( p - mesh.points[pts[0]] ).length(),
            ( p - mesh.points[pts[1]] ).length() };
        std::sort( d, d + 3 );
        const float err = std::abs( dist[v] - d[0] );
        sumErr += err;
        // where the fronts of different starts meet, the distances come along the edges
        if ( d[1] - d[0] > 2 )
            maxErr = std::max( maxErr, err );
    }
    // the fronts meet right from the start points, and the errors near the seams there spread along them
    EXPECT_LT( sumErr / ( n * n ), 3e-2f );
    EXPECT_LT( maxErr, 5e-2f );
}

} //namespace MR
