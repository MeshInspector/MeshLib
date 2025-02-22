#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPolylineTrimWithPlane.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBitSet.h"

namespace MR
{

TEST( MRMesh, TrimPolylineWithPlane )
{
    Polyline3 polyline;
    std::vector<Vector3f> points = { {0.f, 0.f, 0.f}, {2.f, 2.f, 2.f}, {0.f, 4.f, 0.f} };
    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );

    Polyline3 otherPart;
    DividePolylineParameters params;
    params.closeLineAfterCut = true;
    params.otherPart = &otherPart;
    trimWithPlane( polyline, plane, params );

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
    polyline = {};
    otherPart = {};
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


    polyline = {};
    otherPart = {};
    polyline.addFromPoints( points.data(), points.size(), false );
    EdgeId e = polyline.topology.edgePerVertex()[0_v];
    EdgeId e0 = polyline.topology.next( e.sym() );
    for ( ; e0 != e; e0 = polyline.topology.next( e0.sym() ) )
        std::cout << "EdgeId = " << int( e0 ) << " [ " << int( polyline.topology.org( e0 ) ) << " - " << int( polyline.topology.dest( e0 ) ) << " ]\n";



}

}
