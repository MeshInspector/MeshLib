#include <MRCSymbolMesh/MRSymbolMesh.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( int argc, char** argv )
{
    int rc = EXIT_FAILURE;
    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: MeshFromText fontpath text" );
        goto bad_usage;
    }

    MR_SymbolMeshParams* params = MR_SymbolMeshParams_DefaultConstruct();
    MR_SymbolMeshParams_Set_text( params, argv[2], NULL );
    MR_SymbolMeshParams_Set_pathToFontFile( params, argv[1], NULL );

    MR_expected_MR_Mesh_std_string* convRes = MR_createSymbolsMesh( params );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( convRes );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to convert text to mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( convRes ) ) );
        goto fail_conv;
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "mesh.ply", NULL, NULL );
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );

fail_conv:
    MR_expected_MR_Mesh_std_string_Destroy( convRes );
    MR_SymbolMeshParams_Destroy( params );
bad_usage:
    return rc;
}
