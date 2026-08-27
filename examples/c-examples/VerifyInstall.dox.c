#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Build a cube in memory: no input file needed.
    const MR_Vector3f size = MR_Vector3f_diagonal( 1.f );
    const MR_Vector3f base = MR_Vector3f_diagonal( -0.5f );
    MR_Mesh* cube = MR_makeCube( &size, &base );

    const MR_MeshTopology* topology = MR_Mesh_Get_topology( cube );
    printf( "%d verts, %d faces\n",
        MR_MeshTopology_numValidVerts( topology ),
        MR_MeshTopology_numValidFaces( topology ) );

    // Save it, to exercise the file-format code too.
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( cube, "cube.stl", NULL, NULL );
    const MR_std_string* saveErr = MR_expected_void_std_string_error( saveEx );
    if ( saveErr )
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( saveErr ) );
    else
        rc = EXIT_SUCCESS;

    MR_expected_void_std_string_Destroy( saveEx );
    MR_Mesh_Destroy( cube );
    return rc;
}
