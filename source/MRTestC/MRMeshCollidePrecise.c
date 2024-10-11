#include "TestMacros.h"

#include "MRMeshCollidePrecise.h"

#include <MRMeshC/MRAffineXf.h>
#include <MRMeshC/MRContoursCut.h>
#include <MRMeshC/MRIntersectionContour.h>
#include <MRMeshC/MRMatrix3.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshCollidePrecise.h>
#include <MRMeshC/MRTorus.h>
#include <MRMeshC/MRVector3.h>

void testMeshCollidePrecise( void )
{
    MRMakeTorusParameters params = {
        .primaryRadius = 1.1f,
        .secondaryRadius = 0.5f,
        .primaryResolution = 8,
        .secondaryResolution = 8,
    };
    MRMesh* meshA = mrMakeTorus( &params );
    MRMesh* meshB = mrMakeTorus( &params );

    MRVector3f from = mrVector3fPlusZ();
    MRVector3f to = { 0.1f, 0.8f, 0.2f };
    MRMatrix3f rot = mrMatrix3fRotationVector( &from, &to );
    MRAffineXf3f xf = mrAffineXf3fLinear( &rot );
    mrMeshTransform( meshB, &xf, NULL );

    MRMeshPart meshAPart = { meshA, NULL };
    MRMeshPart meshBPart = { meshB, NULL };
    MRCoordinateConverters conv = mrGetVectorConverters( &meshAPart, &meshBPart, NULL );

    MRPreciseCollisionResult* intersections = mrFindCollidingEdgeTrisPrecise( &meshAPart, &meshBPart, conv.toInt, NULL, false );
    const MRVectorEdgeTri edgesAtrisB = mrPreciseCollisionResultEdgesAtrisB( intersections );
    const MRVectorEdgeTri edgesBtrisA = mrPreciseCollisionResultEdgesBtrisA( intersections );
    // FIXME: the results are platform-dependent
    //TEST_ASSERT( edgesAtrisB.size == 80 )
    //TEST_ASSERT( edgesBtrisA.size == 72 )
    TEST_ASSERT( edgesAtrisB.size != 0 )
    TEST_ASSERT( edgesBtrisA.size != 0 )

    const MRMeshTopology* meshATop = mrMeshTopology( meshA );
    const MRMeshTopology* meshBTop = mrMeshTopology( meshB );
    MRContinuousContours* contours = mrOrderIntersectionContours( meshATop, meshBTop, intersections );
    TEST_ASSERT( mrContinuousContoursSize( contours ) == 4 )
    // FIXME: the results are platform-dependent
    //TEST_ASSERT( mrContinuousContoursGet( contours, 0 ).size == 69 )
    //TEST_ASSERT( mrContinuousContoursGet( contours, 1 ).size == 71 )
    //TEST_ASSERT( mrContinuousContoursGet( contours, 2 ).size == 7 )
    //TEST_ASSERT( mrContinuousContoursGet( contours, 3 ).size == 9 )

    MROneMeshContours* meshAContours = mrGetOneMeshIntersectionContours( meshA, meshB, contours, true, &conv, NULL );
    MROneMeshContours* meshBContours = mrGetOneMeshIntersectionContours( meshA, meshB, contours, false, &conv, NULL );
    TEST_ASSERT( mrOneMeshContoursSize( meshAContours ) == 4 )
    TEST_ASSERT( mrOneMeshContoursSize( meshBContours ) == 4 )

    size_t posCount = 0;
    for ( size_t i = 0; i < mrOneMeshContoursSize( meshAContours ); ++i )
        posCount += mrOneMeshContoursGet( meshAContours, i ).intersections.size;
    TEST_ASSERT( posCount == 156 )

    mrOneMeshContoursFree( meshBContours );
    mrOneMeshContoursFree( meshAContours );

    mrContinuousContoursFree( contours );

    mrPreciseCollisionResultFree( intersections );

    mrConvertToFloatVectorFree( conv.toFloat );
    mrConvertToIntVectorFree( conv.toInt );

    mrMeshFree( meshB );
    mrMeshFree( meshA );
}
