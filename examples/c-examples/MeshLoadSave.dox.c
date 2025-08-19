#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRString.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail_load;
    }

    // Save mesh.
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "mesh.ply", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_load:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
