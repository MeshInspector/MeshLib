#include "TestMacros.h"

#include "MRMeshComponents.h"

#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRCube.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMeshPart.h>
#include <MRMeshC/MRMeshBoolean.h>
#include <MRMeshC/MRMeshComponents.h>
#include <MRMeshC/MRVector3.h>

MRMeshPart createMesh(void)
{
    const MRVector3f meshASize = mrVector3fDiagonal( 1.0f );
    const MRVector3f meshABase = mrVector3fDiagonal( -0.5f );

    const MRVector3f meshBSize = mrVector3fDiagonal( 0.1f );
    const MRVector3f meshBBase = mrVector3fDiagonal( 1.0f );

    MRMesh* meshA = mrMakeCube( &meshASize, &meshABase );
    MRMesh* meshB = mrMakeCube( &meshBSize, &meshBBase );

    MRBooleanResult resultAB = mrBoolean( meshA, meshB, MRBooleanOperationUnion, NULL );
    mrMeshFree( meshA );
    mrMeshFree( meshB );

    MRMeshPart mp = { resultAB.mesh, NULL };
    return mp;
}

void testComponentsMap( void )
{
    MRMeshPart mp = createMesh();

    MRMeshComponentsMap map = mrMeshComponentsGetAllComponentsMap( &mp, MRFaceIncidencePerEdge );
    TEST_ASSERT( map.numComponents == 2 );
    TEST_ASSERT( map.faceMap->size == 24 );
    TEST_ASSERT( map.faceMap->data[0].id == 0 );
    TEST_ASSERT( map.faceMap->data[12].id == 1 );

    mrMeshFree( (MRMesh*)mp.mesh );
    mrMeshComponentsAllComponentsMapFree( &map );
}

void testLargeRegions( void )
{
    MRMeshPart mp = createMesh();

    MRMeshComponentsMap map = mrMeshComponentsGetAllComponentsMap( &mp, MRFaceIncidencePerEdge );
    MRMeshRegions regions = mrMeshComponentsGetLargeByAreaRegions( &mp, map.faceMap, map.numComponents, 0.1f );

    TEST_ASSERT( regions.numRegions == 1 );
    TEST_ASSERT( mrBitSetTest( ( const MRBitSet* ) regions.faces, 0 ) );
    TEST_ASSERT( !mrBitSetTest( ( const MRBitSet* ) regions.faces, 12 ) );

    mrMeshFree( ( MRMesh* ) mp.mesh );
    mrMeshComponentsAllComponentsMapFree( &map );
}

void testLargeComponents( void )
{
    MRMeshPart mp = createMesh();

    MRFaceBitSet* components = mrMeshComponentsGetLargeByAreaComponents( &mp, 0.1f, NULL );
    TEST_ASSERT( mrBitSetTest( ( const MRBitSet* ) components, 0 ) );
    TEST_ASSERT( !mrBitSetTest( ( const MRBitSet* ) components, 12 ) );

    mrFaceBitSetFree( components );
    mrMeshFree( ( MRMesh* ) mp.mesh );
}

void testLargestComponent( void )
{
    MRMeshPart mp = createMesh();

    int smallerComponents = 0;
    MRFaceBitSet* largestComponent = mrMeshComponentsGetLargestComponent( &mp, MRFaceIncidencePerEdge, NULL, 0.1f, &smallerComponents );
    TEST_ASSERT( mrBitSetTest( ( const MRBitSet* ) largestComponent, 0 ) );
    TEST_ASSERT( !mrBitSetTest( ( const MRBitSet* ) largestComponent, 12 ) );
    TEST_ASSERT( smallerComponents == 1 );

    mrFaceBitSetFree( largestComponent );
    mrMeshFree( ( MRMesh* ) mp.mesh );
}

void testGetComponent( void )
{
    MRMeshPart mp = createMesh();
    MRFaceId face;
    face.id = 12;
    MRFaceBitSet* component = mrMeshComponentsGetComponent( &mp, face, MRFaceIncidencePerEdge, NULL );
    TEST_ASSERT( !mrBitSetTest( ( const MRBitSet* ) component, 0 ) );
    TEST_ASSERT( mrBitSetTest( ( const MRBitSet* ) component, 12 ) );
    mrFaceBitSetFree( component );
    mrMeshFree( ( MRMesh* ) mp.mesh );
}
