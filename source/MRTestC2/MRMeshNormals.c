#include "TestMacros.h"

#include "MRMeshNormals.h"

#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshNormals.h>

void testMeshNormals( void )
{
    MRVector3f size = mrVector3fDiagonal( 1.f );
    MRVector3f base = mrVector3fDiagonal( -0.5f );
    MRMesh* cube = mrMakeCube( &size, &base );

    MRVertNormals* vertNormals = mrComputePerVertNormals( cube );
    TEST_ASSERT( vertNormals->size == 8 )

    MRFaceNormals* faceNormals = mrComputePerFaceNormals( cube );
    TEST_ASSERT( faceNormals->size == 12 )

    mrVectorVector3fFree( faceNormals );
    mrVectorVector3fFree( vertNormals );
    mrMeshFree( cube );
}
