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

int main( int argc, char** argv )
{
    int rc = EXIT_FAILURE;

    // Both paths are optional: with no arguments this reads mesh.stl and writes mesh.ply
    // in the working directory.
    const char* inputPath = argc > 1 ? argv[1] : "mesh.stl";
    const char* outputPath = argc > 2 ? argv[2] : "mesh.ply";

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( inputPath, NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_value_mut( meshEx );

    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( meshEx ) ) );
        if ( argc <= 1 )
            fprintf( stderr, "Usage: MeshLoadSave [input mesh] [output mesh]\n" );
        goto fail_load;
    }

    // Save mesh.
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, outputPath, NULL, NULL);
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_load:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
