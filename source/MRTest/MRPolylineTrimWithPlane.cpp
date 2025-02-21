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
    polyline.addFromPoints( points.data(), points.size(), true );

    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );

    Polyline3 otherPart;
    DividePolylineParameters params;
    params.closeLineAfterCut = true;
    params.otherPart = &otherPart;
    trimWithPlane( polyline, plane, params );

    EXPECT_TRUE( polyline.topology.isClosed() );
    std::vector<VertId> pointsOnPlane;
    pointsOnPlane.reserve( 2 );
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( polyline.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_TRUE( otherPart.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : otherPart.topology.getValidVerts() )
    {
        if ( otherPart.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );



    polyline = {};
    polyline.addFromPoints( points.data(), points.size(), false );
    otherPart = {};
    trimWithPlane( polyline, plane, params );

    EXPECT_TRUE( polyline.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : polyline.topology.getValidVerts() )
    {
        if ( polyline.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( polyline.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );

    EXPECT_FALSE( otherPart.topology.isClosed() );
    pointsOnPlane.clear();
    for ( VertId v : otherPart.topology.getValidVerts() )
    {
        if ( otherPart.points[v].x == 1.f )
            pointsOnPlane.push_back( v );
    }
    EXPECT_EQ( pointsOnPlane.size(), 2 );
    EXPECT_TRUE( otherPart.topology.findEdge( pointsOnPlane[0], pointsOnPlane[1] ).valid() );
}

}
