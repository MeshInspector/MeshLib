#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPolylineTrimWithPlane.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBitSet.h"

namespace MR
{

TEST( MRMesh, SubdividePolylineWithPlane )
{
    std::vector<Vector3f> points = { {0.f, 0.f, 0.f}, {2.f, 2.f, 2.f}, {4.f, 4.f, 4.f}, {6.f, 6.f, 6.f},
                                     {4.f, 8.f, 4.f}, {2.f, 10.f, 2.f}, {0.f, 12.f, 0.f} };
    Plane3f plane( { 1.f, 0.f, 0.f }, 3.f );
    Polyline3 polyline;
    polyline.addFromPoints( points.data(), points.size(), false );

    EdgeBitSet newPositiveEdges;
    UndirectedEdgeBitSet topUEdges = subdivideWithPlane( polyline, plane, &newPositiveEdges );


    EdgeBitSet expectedNewPositiveEdges;
    expectedNewPositiveEdges.autoResizeSet( 9_e );
    expectedNewPositiveEdges.set( 2_e );

    EXPECT_EQ( newPositiveEdges, expectedNewPositiveEdges );


    UndirectedEdgeBitSet expectedTopUEdges;
    expectedTopUEdges.autoResizeSet( 4_ue );
    expectedTopUEdges.set( 3_ue );
    expectedTopUEdges.set( 2_ue );
    expectedTopUEdges.set( 1_ue );

    EXPECT_EQ( topUEdges, expectedTopUEdges );
}


TEST( MRMesh, TrimPolylineWithPlane )
{
    std::vector<Vector3f> points = { {0.f, 0.f, 0.f}, {2.f, 2.f, 2.f}, {0.f, 4.f, 0.f} };
    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );

    Polyline3 polyline;
    Polyline3 otherPart;
    DividePolylineParameters params;
    params.otherPart = &otherPart;

    auto getPointsOnPlane = []( const Polyline3& polyline )
    {
        std::vector<VertId> pointsOnPlane;
        for ( VertId v : polyline.topology.getValidVerts() )
        {
            if ( polyline.points[v].x == 1.f )
                pointsOnPlane.push_back( v );
        }
        return pointsOnPlane;
    };


    // closed polyline, close after cut
    polyline.addFromPoints( points.data(), points.size(), true );
    params.closeLineAfterCut = true;
    trimWithPlane( polyline, plane, params );

    EXPECT_TRUE( polyline.topology.isClosed() );
    std::vector<VertId> pointsOnPlane = getPointsOnPlane( polyline );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_TRUE( otherPart.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( otherPart );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );


    // closed polyline, do not close after cut
    polyline = {};
    otherPart = {};
    polyline.addFromPoints( points.data(), points.size(), true );
    params.closeLineAfterCut = false;
    trimWithPlane( polyline, plane, params );

    EXPECT_FALSE( polyline.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( polyline );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_FALSE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_FALSE( otherPart.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( otherPart );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_FALSE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );



    // open polyline, close after cut
    polyline = {};
    otherPart = {};
    polyline.addFromPoints( points.data(), points.size(), false );
    params.closeLineAfterCut = true;
    trimWithPlane( polyline, plane, params );

    EXPECT_TRUE( polyline.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( polyline );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_FALSE( otherPart.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( otherPart );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );


    // open polyline, do not close after cut
    polyline = {};
    otherPart = {};
    polyline.addFromPoints( points.data(), points.size(), false );
    params.closeLineAfterCut = false;
    trimWithPlane( polyline, plane, params );

    EXPECT_FALSE( polyline.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( polyline );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_FALSE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_FALSE( otherPart.topology.isClosed() );
    pointsOnPlane = getPointsOnPlane( otherPart );
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_FALSE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

}

}
