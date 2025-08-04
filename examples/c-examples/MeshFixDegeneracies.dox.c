#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshFixer.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRString.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( int argc, char* argv[] )
{
    int rc = EXIT_FAILURE;
    if ( argc != 2 && argc != 3 )
    {
        fprintf( stderr, "Usage: %s INPUT [OUTPUT]", argv[0] );
        return rc;
    }

    const char* input = argv[1];
    const char* output = ( argc == 2 ) ? argv[1] : argv[2];

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( input, NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    // Handle failure to load mesh.
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail_mesh_loading;
    }

    // you can set various parameters for the resolving process; see the documentation for more info
    MR_FixMeshDegeneraciesParams* params = MR_FixMeshDegeneraciesParams_DefaultConstruct();
    // maximum permitted deviation
    const MR_Box3f bbox = MR_Mesh_computeBoundingBox_1( mesh, NULL );
    MR_FixMeshDegeneraciesParams_Set_maxDeviation( params, 1e-5f * MR_Box3f_diagonal( &bbox ) );
    // maximum length of edges to be collapsed
    MR_FixMeshDegeneraciesParams_Set_tinyEdgeLength( params, 1e-3f );

    MR_expected_void_std_string* degeneraciesEx = MR_fixMeshDegeneracies( mesh, params );
    MR_FixMeshDegeneraciesParams_Destroy( params );

    if ( MR_expected_void_std_string_GetError( degeneraciesEx ) )
    {
        fprintf( stderr, "Failed to fix mesh degeneracies: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( degeneraciesEx ) ) );
        goto fail_fix_degen;
    }

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, output, NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_fix_degen:
    MR_expected_void_std_string_Destroy( degeneraciesEx );
fail_mesh_loading:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
