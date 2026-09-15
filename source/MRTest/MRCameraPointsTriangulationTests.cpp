#include "MRMesh/MRCameraPointsTriangulation.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRMeshFixer.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRVector2.h"
#include <gtest/gtest.h>
#include <cmath>

namespace MR
{

TEST( MRMesh, TriangulateCameraPoints )
{
    // integer grid of points on a paraboloid in front of the camera, which is at the origin and looks along +Z
    constexpr int cHalf = 10;
    constexpr int cN = 2 * cHalf + 1;
    VertCoords points;
    for ( int i = -cHalf; i <= cHalf; ++i )
        for ( int j = -cHalf; j <= cHalf; ++j )
            points.emplace_back( float( i ), float( j ), 100 + 0.01f * ( i * i + j * j ) );

    CameraPointsTriangulationSettings settings;
    settings.intrinsics = Matrix3f( { 1000, 0, 500 }, { 0, 1000, 500 }, { 0, 0, 1 } );
    settings.weldPixels = 0;
    auto mesh = triangulateCameraPoints( points, settings );
    ASSERT_TRUE( mesh.has_value() );
    EXPECT_EQ( mesh->points.size(), points.size() );
    EXPECT_EQ( mesh->topology.numValidVerts(), cN * cN );
    EXPECT_EQ( mesh->topology.numValidFaces(), 2 * ( cN - 1 ) * ( cN - 1 ) );
    EXPECT_EQ( mesh->topology.findHoleRepresentiveEdges().size(), 1 );
    for ( FaceId f : mesh->topology.getValidFaces() )
        EXPECT_LT( dot( mesh->normal( f ), mesh->triCenter( f ) ), 0 ); // toward the camera

    // too few points for a triangle: a mesh with the points and without faces
    auto tiny = triangulateCameraPoints( VertCoords{ points[0_v], points[1_v] }, settings );
    ASSERT_TRUE( tiny.has_value() );
    EXPECT_EQ( tiny->points.size(), 2 );
    EXPECT_EQ( tiny->topology.numValidFaces(), 0 );

    // a shifted copy of every point (0.2 px away in the image) is welded into the original vertex, which moves to the average position,
    // and the copy becomes an invalid vertex; vertex ids are the same as point ids
    VertCoords doubled = points;
    for ( const auto & p : points )
        doubled.push_back( p + Vector3f( 0.02f, 0, 0 ) );
    settings.weldPixels = 1;
    VertMap smallestMap;
    settings.outSmallestMap = &smallestMap;
    VertCoords projected;
    settings.outProjectedPoints = &projected;
    auto welded = triangulateCameraPoints( doubled, settings );
    ASSERT_TRUE( welded.has_value() );
    ASSERT_EQ( projected.size(), doubled.size() );
    const auto & p0 = welded->points[0_v];
    EXPECT_LT( ( projected[0_v] - Vector3f( 1000 * p0.x / p0.z + 500, -( 1000 * p0.y / p0.z + 500 ), 1 ) ).length(), 1e-3f );
    settings.outProjectedPoints = nullptr;
    EXPECT_EQ( welded->points.size(), doubled.size() );
    EXPECT_EQ( welded->topology.numValidVerts(), cN * cN );
    EXPECT_EQ( welded->topology.numValidFaces(), 2 * ( cN - 1 ) * ( cN - 1 ) );
    EXPECT_NEAR( welded->points[0_v].x, points[0_v].x + 0.01f, 1e-5f );
    ASSERT_EQ( smallestMap.size(), doubled.size() );
    const VertId copies( int( points.size() ) );
    for ( VertId v( 0 ); v < points.size(); ++v )
    {
        EXPECT_TRUE( welded->topology.hasVert( v ) );
        EXPECT_FALSE( welded->topology.hasVert( v + copies ) );
        EXPECT_EQ( smallestMap[v], v );
        EXPECT_EQ( smallestMap[v + copies], v );
    }

    // the same points as a cloud with the copies invalid: welding has nothing to merge, and the moving overload empties the cloud
    PointCloud cloud;
    cloud.points = doubled;
    cloud.validPoints.resize( doubled.size() );
    for ( VertId v( 0 ); v < points.size(); ++v )
        cloud.validPoints.set( v );
    auto fromCloud = triangulateCameraPoints( cloud, settings );
    ASSERT_TRUE( fromCloud.has_value() );
    EXPECT_EQ( fromCloud->points.size(), doubled.size() );
    EXPECT_EQ( fromCloud->topology.numValidVerts(), cN * cN );
    EXPECT_EQ( fromCloud->topology.numValidFaces(), 2 * ( cN - 1 ) * ( cN - 1 ) );
    EXPECT_EQ( fromCloud->points[0_v], points[0_v] );
    EXPECT_FALSE( fromCloud->topology.hasVert( copies ) );
    EXPECT_EQ( cloud.points.size(), doubled.size() );
    auto fromMovedCloud = triangulateCameraPoints( std::move( cloud ), settings );
    ASSERT_TRUE( fromMovedCloud.has_value() );
    EXPECT_EQ( fromMovedCloud->topology.numValidFaces(), fromCloud->topology.numValidFaces() );
    EXPECT_TRUE( cloud.points.empty() );

    // without the points inside radius 3 the Delaunay bridges the gap, and deleteFacesWithLongEdges reopens it as a hole
    VertCoords holed;
    for ( const auto & p : points )
        if ( sqr( p.x ) + sqr( p.y ) >= 9 )
            holed.push_back( p );
    auto bridged = triangulateCameraPoints( holed, settings );
    ASSERT_TRUE( bridged.has_value() );
    EXPECT_EQ( bridged->topology.numValidVerts(), holed.size() );
    EXPECT_EQ( bridged->topology.findHoleRepresentiveEdges().size(), 1 );
    Mesh open = *bridged;
    deleteFacesWithLongEdges( open, 1.5f );
    EXPECT_EQ( open.topology.findHoleRepresentiveEdges().size(), 2 );
    EXPECT_LT( open.topology.numValidFaces(), bridged->topology.numValidFaces() );
    for ( UndirectedEdgeId ue : undirectedEdges( open.topology ) )
        EXPECT_LE( open.edgeLength( ue ), 1.5f );
}

TEST( MRMesh, SmoothCameraMeshDepth )
{
    // constant depth (a harmonic field even at the grid boundary) sampled on a grid with a deterministic depth noise;
    // the smoothing must bring the vertices closer to the plane while keeping their projections
    constexpr int cHalf = 10;
    VertCoords exact, noisy;
    for ( int i = -cHalf; i <= cHalf; ++i )
        for ( int j = -cHalf; j <= cHalf; ++j )
        {
            const float z = 100;
            exact.emplace_back( float( i ), float( j ), z );
            const float noise = 0.1f * ( ( ( i * 7 + j * 13 ) % 5 + 5 ) % 5 - 2 ); // in [-0.2, 0.2]
            noisy.push_back( exact.back() * ( ( z + noise ) / z ) );      // along the viewing ray
        }
    CameraPointsTriangulationSettings settings;
    settings.intrinsics = Matrix3f( { 1000, 0, 500 }, { 0, 1000, 500 }, { 0, 0, 1 } );
    settings.weldPixels = 0;
    auto mesh = triangulateCameraPoints( noisy, settings );
    ASSERT_TRUE( mesh.has_value() );
    ASSERT_EQ( mesh->points.size(), exact.size() );

    auto rmsError = [&]( const VertCoords & pts )
    {
        double sum = 0;
        for ( VertId v( 0 ); v < pts.size(); ++v )
            sum += sqr( pts[v].z - exact[v].z );
        return std::sqrt( sum / pts.size() );
    };
    const auto errBefore = rmsError( mesh->points );
    for ( auto edgeWeights : { EdgeWeights::Unit, EdgeWeights::Cotan } )
        for ( auto type : { AreaStabilizer::Uniform, AreaStabilizer::Area, AreaStabilizer::AreaSq } )
        {
            Mesh smoothed = *mesh;
            smoothCameraMeshDepth( smoothed, { .edgeWeights = edgeWeights, .bdStabilizer = 0.1f, .innerStabilizer = 0.1f, .innerStabilizerType = type } );
            EXPECT_LT( rmsError( smoothed.points ), 0.5 * errBefore );
            for ( VertId v( 0 ); v < exact.size(); ++v )
            {
                const auto & p = smoothed.points[v];
                EXPECT_LT( ( Vector2f( p.x / p.z, p.y / p.z ) - Vector2f( exact[v].x / exact[v].z, exact[v].y / exact[v].z ) ).length(), 1e-6f );
            }
        }

    // with the default settings the boundary vertices are attracted to their noisy depths much stronger than inner ones
    Mesh smoothed = *mesh;
    smoothCameraMeshDepth( smoothed );
    VertCoords points2 = mesh->points;
    smoothCameraMeshDepth( mesh->topology, points2 );
    EXPECT_EQ( points2, smoothed.points );
    double bdMove = 0, innerMove = 0;
    int nBd = 0, nInner = 0;
    for ( VertId v( 0 ); v < exact.size(); ++v )
    {
        const auto move = std::abs( smoothed.points[v].z - mesh->points[v].z );
        if ( smoothed.topology.isBdVertex( v ) )
        {
            bdMove += move;
            ++nBd;
        }
        else
        {
            innerMove += move;
            ++nInner;
        }
    }
    EXPECT_LT( bdMove / nBd, 0.5 * innerMove / nInner );
}

} //namespace MR
