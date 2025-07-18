#include "TestMacros.h"

#include "MRMeshNormals.h"

#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshNormals.h>
#include <MRCMesh/MRVector.h>

void testMeshNormals( void )
{
    MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );
    MR_Mesh* cube = MR_makeCube( &size, &base );

    MR_VertCoords* vertNormals = MR_computePerVertNormals( cube );
    TEST_ASSERT( MR_VertCoords_size( vertNormals ) == 8 )

    MR_FaceNormals* faceNormals = MR_computePerFaceNormals( cube );
    TEST_ASSERT( MR_FaceNormals_size( faceNormals ) == 12 )

    MR_FaceNormals_Destroy( faceNormals );
    MR_VertCoords_Destroy( vertNormals );
    MR_Mesh_Destroy( cube );
}
