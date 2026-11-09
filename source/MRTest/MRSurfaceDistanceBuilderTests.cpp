#include <MRMesh/MRSurfaceDistanceBuilder.h>
#include <MRMesh/MRSurfaceDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRConstants.h>
#include <MRMesh/MRLineSegm.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshTriPoint.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRphmap.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>

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

TEST( MRMesh, SurfaceDistanceAdjacentPoints )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true );
    const VertId s0( n / 2 * n + n / 2 ), s1( s0 + 1 );
    VertBitSet points( mesh.topology.vertSize() );
    points.set( s0 );
    points.set( s1 );
    // two separate start points at the ends of one edge: the fronts meet right from the start
    const auto dist = computeSurfaceDistances( mesh, UndirectedEdgeBitSet{}, points );
    float sumErr = 0, maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const float err = std::abs( dist[v] - std::min( ( mesh.points[v] - mesh.points[s0] ).length(), ( mesh.points[v] - mesh.points[s1] ).length() ) );
        sumErr += err;
        maxErr = std::max( maxErr, err );
    }
    EXPECT_LT( sumErr / ( n * n ), 1e-2f );
    EXPECT_LT( maxErr, 0.2f );
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
    HashMap<VertId, float> lineStarts;
    for ( auto v : lineVerts )
        lineStarts[v] = 0;
    EXPECT_TRUE( computeSurfaceDistances( mesh, lineEdges, VertBitSet{} ) == computeSurfaceDistances( mesh, lineStarts ) );

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
    EXPECT_LT( sumErr / ( n * n ), 3.5e-2f );
    EXPECT_LT( maxErr, 5e-2f );
}

TEST( MRMesh, SurfaceDistanceEdgeMiddle )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, false );
    const VertId v0( n / 2 * n + n / 2 );
    // a start in the middle of an axis-aligned edge and of a diagonal edge
    for ( VertId v1 : { v0 + 1, v0 + n + 1 } )
    {
        const MeshTriPoint start( MeshEdgePoint( mesh.topology.findEdge( v0, v1 ), 0.5f ) );
        const auto p = mesh.triPoint( start );
        const auto dist = computeSurfaceDistances( mesh, start );
        float maxErr = 0;
        for ( auto v : mesh.topology.getValidVerts() )
            maxErr = std::max( maxErr, std::abs( dist[v] - ( mesh.points[v] - p ).length() ) );
        EXPECT_LT( maxErr, 1e-3f );
    }
}

TEST( MRMesh, SurfaceDistanceTriPointStarts )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true );
    std::vector<MeshTriPoint> starts;
    std::vector<Vector3f> ps;
    for ( VertId v : { VertId( n / 2 * n + 10 ), VertId( n / 2 * n + 30 ), VertId( ( n / 2 + 2 ) * n + 20 ) } )
    {
        starts.push_back( MeshTriPoint( mesh.topology.findEdge( v, v + 1 ), { 0.3f, 0.3f } ) );
        ps.push_back( mesh.triPoint( starts.back() ) );
    }
    const auto dist = computeSurfaceDistances( mesh, starts );
    float sumErr = 0, maxErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        float d = FLT_MAX;
        for ( const auto & p : ps )
            d = std::min( d, ( mesh.points[v] - p ).length() );
        const float err = std::abs( dist[v] - d );
        sumErr += err;
        maxErr = std::max( maxErr, err );
    }
    EXPECT_LT( sumErr / ( n * n ), 2e-3f );
    EXPECT_LT( maxErr, 0.2f );
}

TEST( MRMesh, SurfaceDistanceRingWithGap )
{
    const int n = 41;
    const auto mesh = makePlanarGrid( n, true );
    // the perimeter of a square of start vertices without one vertex: the ends of one line meet at the gap
    VertBitSet seeds( mesh.topology.vertSize() );
    for ( int i = 10; i <= 30; ++i )
        for ( VertId v : { VertId( 10 * n + i ), VertId( 30 * n + i ), VertId( i * n + 10 ), VertId( i * n + 30 ) } )
            seeds.set( v );
    seeds.reset( VertId( 30 * n + 20 ) );
    const auto edges = getInnerEdges( mesh.topology, seeds );
    const auto dist = computeSurfaceDistances( mesh, seeds );
    float sumErr = 0, minErrNearGap = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const auto & p = mesh.points[v];
        float d = FLT_MAX;
        for ( auto s : seeds )
            d = std::min( d, ( p - mesh.points[s] ).length() );
        for ( auto ue : edges )
            d = std::min( d, ( p - closestPointOnLineSegm( p, LineSegm3f( mesh.points[mesh.topology.org( ue )], mesh.points[mesh.topology.dest( ue )] ) ) ).length() );
        sumErr += std::abs( dist[v] - d );
        if ( std::abs( p.x - 20 ) <= 4 && std::abs( p.y - 30 ) <= 4 )
            minErrNearGap = std::min( minErrNearGap, dist[v] - d );
    }
    EXPECT_LT( sumErr / ( n * n ), 4e-2f );
    EXPECT_GT( minErrNearGap, -0.1f );
}

TEST( MRMesh, SurfaceDistanceVaryingStartValues )
{
    const int n = 41;
    const int row = n / 2;
    const float k = 0.5f;
    const auto mesh = makePlanarGrid( n, true, row );
    // a straight row of start vertices with the start values growing along it
    HashMap<VertId, float> starts;
    for ( int x = 5; x <= 35; ++x )
        starts[VertId( row * n + x )] = k * ( x - 5 );
    const auto dist = computeSurfaceDistances( mesh, starts );
    float sumErr = 0, minErr = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const auto & p = mesh.points[v];
        const float h = std::abs( p.y - row );
        const float t = std::clamp( p.x - k * h / std::sqrt( 1 - k * k ), 5.f, 35.f );
        const float err = dist[v] - ( k * ( t - 5 ) + std::hypot( p.x - t, h ) );
        sumErr += std::abs( err );
        minErr = std::min( minErr, err );
    }
    EXPECT_LT( sumErr / ( n * n ), 0.1f );
    EXPECT_GT( minErr, -1e-3f );
}

// the distances on a unit sphere mesh against the exact geodesic distances on the sphere:
// the mean error, and the largest overestimate and underestimate
struct SphereErrors
{
    float mean = 0, over = 0, under = 0;
};

static SphereErrors sphereErrors( const Mesh & mesh, const VertScalars & dist, const std::function<float( const Vector3f & )> & exact )
{
    SphereErrors res;
    double sum = 0;
    for ( auto v : mesh.topology.getValidVerts() )
    {
        const float err = dist[v] - exact( mesh.points[v] );
        sum += std::abs( err );
        res.over = std::max( res.over, err );
        res.under = std::max( res.under, -err );
    }
    res.mean = float( sum / mesh.topology.numValidVerts() );
    return res;
}

static float sphereAngle( const Vector3f & a, const Vector3f & b )
{
    return std::atan2( cross( a, b ).length(), dot( a, b ) );
}

TEST( MRMesh, SurfaceDistanceSpherePoints )
{
    const auto mesh = makeSphere( { .radius = 1, .numMeshVertices = 10000 } );
    const VertId s0( 0 );
    const auto p0 = mesh.points[s0];
    // the vertex at about 90 degrees from s0
    VertId s1;
    for ( auto v : mesh.topology.getValidVerts() )
        if ( !s1 || std::abs( dot( mesh.points[v], p0 ) ) < std::abs( dot( mesh.points[s1], p0 ) ) )
            s1 = v;
    const auto p1 = mesh.points[s1];
    const MeshTriPoint start( mesh.topology.edgeWithLeft( FaceId( 0 ) ), { 0.3f, 0.3f } );
    const auto ps = mesh.triPoint( start );

    VertBitSet seeds( mesh.topology.vertSize() );
    seeds.set( s0 );
    const auto one = computeSurfaceDistances( mesh, seeds );
    seeds.set( s1 );
    const auto two = computeSurfaceDistances( mesh, seeds );
    for ( const auto & errs : {
        sphereErrors( mesh, one, [&]( const Vector3f & p ) { return sphereAngle( p, p0 ); } ),
        sphereErrors( mesh, two, [&]( const Vector3f & p ) { return std::min( sphereAngle( p, p0 ), sphereAngle( p, p1 ) ); } ),
        sphereErrors( mesh, computeSurfaceDistances( mesh, start ), [&]( const Vector3f & p ) { return sphereAngle( p, ps ); } ) } )
    {
        EXPECT_LT( errs.mean, 3e-3f );
        EXPECT_LT( errs.over, 1e-3f );
        // the paths on the mesh are shorter than on the sphere, most of all where the fronts meet
        EXPECT_LT( errs.under, 3e-2f );
    }
}

TEST( MRMesh, SurfaceDistanceSphereLines )
{
    const int nLon = 64, nLat = 35;
    const auto mesh = makeUVSphere( 1, nLon, nLat );
    auto latitude = [&]( int j ) { return PI_F * ( float( j + 1 ) / ( nLat + 1 ) - 0.5f ); };
    auto pointLatitude = []( const Vector3f & p ) { return std::asin( std::clamp( p.z, -1.f, 1.f ) ); };

    // the parallel at 30 degrees: the line is curved on the sphere
    const int j30 = 23;
    VertBitSet seeds( mesh.topology.vertSize() );
    for ( int i = 0; i < nLon; ++i )
        seeds.set( VertId( j30 * nLon + i ) );
    const auto parallel = sphereErrors( mesh, computeSurfaceDistances( mesh, seeds ),
        [&]( const Vector3f & p ) { return std::abs( pointLatitude( p ) - latitude( j30 ) ); } );
    EXPECT_LT( parallel.mean, 1e-3f );
    EXPECT_LT( parallel.over, 1e-3f );
    EXPECT_LT( parallel.under, 3e-3f );

    // a quarter of the equator: the distances from its ends are on both sides
    const int jEq = nLat / 2;
    seeds.reset();
    for ( int i = 0; i <= nLon / 4; ++i )
        seeds.set( VertId( jEq * nLon + i ) );
    const auto a = mesh.points[VertId( jEq * nLon )], b = mesh.points[VertId( jEq * nLon + nLon / 4 )];
    const auto arc = sphereErrors( mesh, computeSurfaceDistances( mesh, seeds ), [&]( const Vector3f & p )
    {
        const float lon = std::atan2( p.x, p.y );
        if ( lon >= 0 && lon <= PI_F / 2 )
            return std::abs( pointLatitude( p ) );
        return std::min( sphereAngle( p, a ), sphereAngle( p, b ) );
    } );
    EXPECT_LT( arc.mean, 4e-3f );
    EXPECT_LT( arc.over, 1e-3f );
    EXPECT_LT( arc.under, 2e-2f );
}

} //namespace MR
