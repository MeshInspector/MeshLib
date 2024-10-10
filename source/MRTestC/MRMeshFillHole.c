#include "TestMacros.h"

#include "MRMeshFillHole.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshFillHole.h>
#include <MRMeshC/MRMeshTopology.h>

void testMeshFillHole( void )
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
    MRMesh* mesh = mrMeshFromTriangles( points, 6, (const MRThreeVertIds*)t, 5 );

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
