#include <MRCIOExtras/MRCtm.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_value_mut( meshEx );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( meshEx ) ) );
        goto fail_load;
    }

    // Save mesh
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat( mesh, "mesh.ply", NULL, NULL );
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
        goto fail_save;
    }

    // More mesh formats are available in the MRIOExtras library
    MR_expected_void_std_string* saveCtmEx = MR_MeshSave_toCtm_const_std_filesystem_path_ref_MR_SaveSettings( mesh, "mesh.ctm", NULL, NULL );
    if ( MR_expected_void_std_string_error( saveCtmEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveCtmEx ) ) );
        goto fail_save_ctm;
    }

    rc = EXIT_SUCCESS;
fail_save_ctm:
    MR_expected_void_std_string_Destroy( saveCtmEx );
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_load:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
