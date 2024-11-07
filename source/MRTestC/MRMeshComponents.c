#include "TestMacros.h"

#include "MRMeshComponents.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshPart.h>
#include <MRMeshC/MRMeshBoolean.h>
#include <MRMeshC/MRMeshComponents.h>
#include <MRMeshC/MRVector3.h>

void testComponentsMap( void )
{
    const MRVector3f meshSize = mrVector3fDiagonal( 1.0f );
    const MRVector3f meshABase = mrVector3fDiagonal( -0.5f );
    const MRVector3f meshBBase = mrVector3fDiagonal( -1.5f );

    MRMesh* meshA = mrMakeCube( &meshSize, &meshABase );
    MRMesh* meshB = mrMakeCube( &meshSize, &meshBBase );

    MRBooleanResult resultAB = mrBoolean( meshA, meshB, MRBooleanOperationUnion, NULL );
    MRMeshPart mp = { resultAB.mesh, NULL };

    MRMeshComponentsMap map = mrMeshComponentsGetAllComponentsMap( &mp, MRFaceIncidencePerEdge );
    TEST_ASSERT(map.numComponents == 2);
    TEST_ASSERT( map.faceMap->size == 24 );
    TEST_ASSERT( map.faceMap->data[0].id == 0 );
    TEST_ASSERT( map.faceMap->data[12].id == 1 );

    mrMeshFree( meshA );
    mrMeshFree( meshB );
    mrMeshFree( resultAB.mesh );
    mrMeshComponentsAllComponentsMapFree( &map );
}

void testLargeRegions( void )
{
    const MRVector3f meshASize = mrVector3fDiagonal( 1.0f );
    const MRVector3f meshABase = mrVector3fDiagonal( -0.5f );

    const MRVector3f meshBSize = mrVector3fDiagonal( 0.1f );
    const MRVector3f meshBBase = mrVector3fDiagonal( 1.0f );

    MRMesh* meshA = mrMakeCube( &meshASize, &meshABase );
    MRMesh* meshB = mrMakeCube( &meshBSize, &meshBBase );

    MRBooleanResult resultAB = mrBoolean( meshA, meshB, MRBooleanOperationUnion, NULL );
    MRMeshPart mp = { resultAB.mesh, NULL };

    MRMeshComponentsMap map = mrMeshComponentsGetAllComponentsMap( &mp, MRFaceIncidencePerEdge );
    MRMeshRegions regions = mrMeshComponentsGetLargeByAreaRegions( &mp, map.faceMap, map.numComponents, 0.1f );

    TEST_ASSERT( regions.numRegions == 1 );
    TEST_ASSERT( mrBitSetTest( (const MRBitSet*)regions.faces, 0 ) );
    TEST_ASSERT( !mrBitSetTest( ( const MRBitSet* ) regions.faces, 12 ) );

    mrMeshFree( meshA );
    mrMeshFree( meshB );
    mrMeshFree( resultAB.mesh );
    mrMeshComponentsAllComponentsMapFree( &map );
}
