#include "MRMesh/MRCameraPointsTriangulation.h"
#include "MRMesh/MRMesh.h"
#include "MRMesh/MREdgeIterator.h"
#include "MRMesh/MRMeshFixer.h"
#include <gtest/gtest.h>

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
    EXPECT_EQ( mesh->topology.numValidVerts(), cN * cN );
    EXPECT_EQ( mesh->topology.numValidFaces(), 2 * ( cN - 1 ) * ( cN - 1 ) );
    EXPECT_EQ( mesh->topology.findHoleRepresentiveEdges().size(), 1 );
    for ( FaceId f : mesh->topology.getValidFaces() )
        EXPECT_LT( dot( mesh->normal( f ), mesh->triCenter( f ) ), 0 ); // toward the camera

    // a shifted copy of every point (0.2 px away in the image) is welded back into one vertex at the average position
    VertCoords doubled = points;
    for ( const auto & p : points )
        doubled.push_back( p + Vector3f( 0.02f, 0, 0 ) );
    settings.weldPixels = 1;
    auto welded = triangulateCameraPoints( doubled, settings );
    ASSERT_TRUE( welded.has_value() );
    EXPECT_EQ( welded->topology.numValidVerts(), cN * cN );
    EXPECT_EQ( welded->topology.numValidFaces(), 2 * ( cN - 1 ) * ( cN - 1 ) );
    EXPECT_NEAR( welded->points[0_v].x, points[0_v].x + 0.01f, 1e-5f );

    // without the points inside radius 3 the Delaunay bridges the gap, and the edge-length limit reopens it as a hole
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

} //namespace MR
