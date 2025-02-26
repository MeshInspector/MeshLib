#include "MRMesh/MRGTest.h"
#include "MRMesh/MRMeshFwd.h"
#include "MRMesh/MRPolylineTrimWithPlane.h"
#include "MRMesh/MRPolyline.h"
#include "MRMesh/MRPlane3.h"
#include "MRMesh/MRBitSet.h"

namespace MR
{

TEST( MRMesh, ExtractSectionsFromPolyline )
{
    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );

    auto check = [&plane] ( const std::vector<Vector3f>& points, float eps, std::vector<EdgeSegment> edgeSegmentsOut, UndirectedEdgeBitSet positiveEdgesOut )
    {
        Polyline3 polyline;
        polyline.addFromPoints( points.data(), points.size(), false );
        UndirectedEdgeBitSet positiveEdgesIn;
        std::vector<EdgeSegment> edgeSegmentsIn = extractSectionsFromPolyline( polyline, plane, eps, &positiveEdgesIn );
        EXPECT_EQ( edgeSegmentsIn, edgeSegmentsOut );
        EXPECT_EQ( positiveEdgesIn, positiveEdgesOut );
    };

    UndirectedEdgeBitSet bsOne;
    bsOne.autoResizeSet( 0_ue );

    // eps == 0.f
    std::vector<std::vector<std::vector<EdgeSegment>>> expectedEdgeSegments( 3 );
    std::vector<std::vector<UndirectedEdgeBitSet>> expectedPositiveEdges( 3 );

    expectedEdgeSegments[0] = std::vector<std::vector<EdgeSegment>>{ {}, { {0_e, 1.f, 1.f} }, { {0_e, 0.5f, 0.5f} } };
    expectedEdgeSegments[1] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 1.f, 1.f} }, {}, {} };
    expectedEdgeSegments[2] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 0.5f, 0.5f} }, {}, {} };

    expectedPositiveEdges[0] = { {}, {}, {} };
    expectedPositiveEdges[1] = { {}, bsOne, bsOne };
    expectedPositiveEdges[2] = { {}, bsOne, bsOne };

    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 3; ++j )
            check( { {float( i ), 0.f, 0.f}, {float( j ), 1.f, 0.f} }, 0.f, expectedEdgeSegments[i][j], expectedPositiveEdges[i][j] );
    }


    // eps > 0.f
    plane = Plane3f( { 1.f, 0.f, 0.f }, 2.f );

    expectedEdgeSegments.resize( 5 );
    expectedPositiveEdges.resize( 5 );

    expectedEdgeSegments[0] = std::vector<std::vector<EdgeSegment>>{ {}, { {0_e, 1.f, 1.f} }, { {0_e, 0.5f, 1.f} }, { {0_e, 1 / 3.f, 1.f} }, { {0_e, 0.25f, 0.75f} } };
    expectedEdgeSegments[1] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 1.f, 1.f} }, {}, { {0_e, 0.f, 1.f} }, { {0_e, 0.f, 1.f} }, { {0_e, 0.f, 1 - 1 / 3.f} } };
    expectedEdgeSegments[2] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 0.5f, 1.f} }, { {1_e, 0.f, 1.f} }, { {0_e, 0.f, 1.f} }, { {0_e, 0.f, 1.f} }, { {0_e, 0.f, 0.5f} } };
    expectedEdgeSegments[3] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 1 / 3.f, 1.f} }, { {1_e, 0.f, 1.f} }, { {1_e, 0.f, 1.f} }, {}, {} };
    expectedEdgeSegments[4] = std::vector<std::vector<EdgeSegment>>{ { {1_e, 0.25f, 0.75f} }, { {1_e, 0.f, 1 - 1 / 3.f} }, { {1_e, 0.0f, 0.5f} }, {}, {} };

    expectedPositiveEdges[0] = { {}, {}, {}, {}, {} };
    for ( int i = 1; i < 5; ++i )
        expectedPositiveEdges[i] = { {}, bsOne, bsOne, bsOne, bsOne };

    for ( int i = 0; i < 5; ++i )
    {
        for ( int j = 0; j < 5; ++j )
            check( { {float( i ), 0.f, 0.f}, {float( j ), 1.f, 0.f} }, 1.f, expectedEdgeSegments[i][j], expectedPositiveEdges[i][j] );
    }
}

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

TEST( MRMesh, TrimWithPlaneInfinity )
{
    // infinity cycle
    std::vector<Vector3f> points = { {0.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {2.f, 0.f, 0.f} };
    Plane3f plane( { 1.f, 0.f, 0.f }, 1.f );
    Polyline3 polyline;
    polyline.addFromPoints( points.data(), points.size(), true );
    trimWithPlane( polyline, plane );
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
