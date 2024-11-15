#include "TestMacros.h"

#include "MRMeshFillHole.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshFillHole.h>
#include <MRMeshC/MRFillHoleNicely.h>
#include <MRMeshC/MRRegionBoundary.h>
#include <MRMeshC/MRMeshTopology.h>
#include <MRMeshC/MRMeshFixer.h>

MRMesh* createMeshWithHoles( void )
{
    const MRVector3f points[] = {
        { 0.f, 0.f, 0.f },
        { 1.f, 0.f, 0.f },
        { 0.f, 1.f, 0.f },
        { 0.f, 0.f, 1.f },
        { 1.f, 0.f, 1.f },
        { 0.f, 1.f, 1.f },
    };
    const int t[] = {
        0, 2, 1,
        3, 4, 5,
        0, 1, 3,
        2, 5, 4,
        2, 3, 5,
    };

    return mrMeshFromTriangles( points, 6, ( const MRThreeVertIds* ) t, 5 );
}

void testMeshFillHole( void )
{
    MRMesh* mesh = createMeshWithHoles();

    MREdgePath* oldBdEdges = mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
    TEST_ASSERT( oldBdEdges->size == 2 );

    MRFillHoleParams params = mrFillHoleParamsNew();
    MRFaceBitSet* newFaces = mrFaceBitSetNew( 0, false );
    params.outNewFaces = newFaces;
    mrFillHoles( mesh, oldBdEdges->data, oldBdEdges->size, &params );

    TEST_ASSERT( mrBitSetCount( (MRBitSet*)newFaces ) == 3 )

    MREdgePath* newBdEdges = mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
    TEST_ASSERT( newBdEdges->size == 0 );

    mrEdgePathFree( newBdEdges );
    mrFaceBitSetFree( newFaces );
    mrEdgePathFree( oldBdEdges );
    mrMeshFree( mesh );
}

void testMeshFillHoleNicely( void )
{
    MRMesh* mesh = createMeshWithHoles();
    MREdgePath* oldBdEdges = mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
    TEST_ASSERT( oldBdEdges->size == 2 );

    MRFillHoleNicelyParams params = mrFillHoleNicelyParamsNew();

    MRFaceBitSet* patch = mrFillHoleNicely( mesh, oldBdEdges->data[0], &params );
    size_t patchCount = mrBitSetCount( (MRBitSet*)patch );
    TEST_ASSERT( patchCount == 1887 );

    MREdgePath* newBdEdges = mrMeshTopologyFindHoleRepresentiveEdges( mrMeshTopology( mesh ) );
    TEST_ASSERT( newBdEdges->size == 1 );

    mrFaceBitSetFree( patch );
    mrEdgePathFree( oldBdEdges );
    mrEdgePathFree( newBdEdges );
    mrMeshFree( mesh );
}

void testRightBoundary( void )
{
    MRMesh* mesh = createMeshWithHoles();
    MREdgeLoops* loops = mrFindRightBoundary( mrMeshTopology( mesh ), NULL );

    TEST_ASSERT( mrEdgeLoopsSize( loops ) == 2 );
    MREdgeLoop loop = mrEdgeLoopsGet( loops, 0 );
    TEST_ASSERT( loop.size == 3 );

    loop = mrEdgeLoopsGet( loops, 1 );
    TEST_ASSERT( loop.size == 4 );

    mrEdgeLoopsFree( loops );
    mrMeshFree( mesh );
}

void testFindHoleComplicatingFaces( void )
{
    MRMesh* mesh = createMeshWithHoles();
    MRFaceBitSet* faces = mrFindHoleComplicatingFaces( mesh );
    const size_t facesCount = mrBitSetCount( (MRBitSet*)faces );
    TEST_ASSERT( facesCount == 0 );
    mrFaceBitSetFree( faces );
    mrMeshFree( mesh );
}
