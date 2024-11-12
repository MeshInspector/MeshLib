#include "TestMacros.h"

#include "MRMesh.h"
#include "MRMeshC/MRMesh.h"
#include "MRMeshC/MRCube.h"
#include "MRMeshC/MRBitSet.h"

void testArea( void )
{
    const MRVector3f size = mrVector3fDiagonal( 1.f );
    const MRVector3f base = mrVector3fDiagonal( -0.5f );

    MRMesh* mesh = mrMakeCube( &size, &base );
    

    double area = mrMeshArea( mesh, NULL );
    TEST_ASSERT( area > 5.999f && area < 6.001f );

    MRFaceBitSet* faces = mrFaceBitSetNew( 12, true );
    for ( int i = 6; i < 12; ++i )
        mrBitSetSet( (MRBitSet*)faces, i, false );

    area = mrMeshArea( mesh, faces );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    mrMeshDeleteFaces( mesh, faces, NULL );

    area = mrMeshArea( mesh, NULL );
    TEST_ASSERT( area > 2.999f && area < 3.001f );

    mrFaceBitSetFree( faces );
    mrMeshFree( mesh );    
}