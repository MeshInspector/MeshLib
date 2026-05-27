#include <MRMesh/MRPolylineDecimate.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRVector2.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, DecimatePolyline )
{
    std::vector< Contour2f> testContours;
    // rhombus
    Contour2f contRhombus;
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    contRhombus.push_back( Vector2f( 1.f, 1.f ) );
    contRhombus.push_back( Vector2f( 2.f, 1.f ) );
    contRhombus.push_back( Vector2f( 2.f, 0.f ) );
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    testContours.push_back( contRhombus );

    // square
    Contour2f contSquare;
    contRhombus.push_back( Vector2f( 0.f, 0.f ) );
    contRhombus.push_back( Vector2f( 0.f, 1.f ) );
    contRhombus.push_back( Vector2f( 1.f, 1.f ) );
    contRhombus.push_back( Vector2f( 1.f, 0.f ) );
    testContours.push_back( contRhombus );

    // square with self-intersections
    Contour2f contSquareSelfIntersected;
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 0.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 1.f, 1.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 1.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 1.f, 0.f ) );
    contSquareSelfIntersected.push_back( Vector2f( 0.f, 0.f ) );
    testContours.push_back( contSquareSelfIntersected );

    // simple small line
    Contour2f smallLine;
    smallLine.push_back( Vector2f( 0.f, 0.f ) );
    smallLine.push_back( Vector2f( 1.f, 1.f ) );
    smallLine.push_back( Vector2f( 2.f, 2.f ) );
    testContours.push_back( smallLine );

    // arc
    Contour2f contArc;
    contArc.push_back( Vector2f( -2.f, 0.f ) );
    contArc.push_back( Vector2f( -1.f, 1.f ) );
    contArc.push_back( Vector2f( 0.f, 1.5f ) );
    contArc.push_back( Vector2f( 1.f, 1.f ) );
    contArc.push_back( Vector2f( 2.f, 0.f ) );
    testContours.push_back( contArc );

    for( auto& cont : testContours )
    {
        DecimatePolylineSettings2 settings;
        settings.maxDeletedVertices = 3;
        settings.maxError = 100.f;
        settings.touchBdVertices = false;

        MR::Polyline2 pl( { cont } );
        auto plBack = pl;
        auto decRes = decimatePolyline( pl, settings );

        int validLines = 0;
        for ( UndirectedEdgeId ue{0}; ue < pl.topology.undirectedEdgeSize(); ++ue )
            if ( !pl.topology.isLoneEdge( ue ) )
                ++validLines;

        EXPECT_TRUE( validLines > 0 );
        EXPECT_EQ( validLines + decRes.vertsDeleted, plBack.topology.undirectedEdgeSize() );
    }
}

} //namespace MR
