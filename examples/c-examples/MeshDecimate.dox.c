#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshDecimate.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRString.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    // Handle failure to load mesh.
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail;
    }

    // Setup decimate parameters
    MR_DecimateSettings* params = MR_DecimateSettings_DefaultConstruct();

    // Decimation stop thresholds, you may specify one or both
    MR_DecimateSettings_Set_maxDeletedFaces( params, 1000 ); // Number of faces to be deleted
    MR_DecimateSettings_Set_maxError( params, 0.05f ); // Maximum error when decimation stops

    // Number of parts to simultaneous processing, greatly improves performance by cost of minor quality loss.
    // Recommended to set to the number of available CPU cores or more for the best performance
    MR_DecimateSettings_Set_subdivideParts( params, 64 );

    // Decimate mesh
    MR_DecimateResult* result = MR_decimateMesh( mesh, params );
    MR_DecimateSettings_Destroy( params );

    printf( "Removed %d vertices, %d faces\n", *MR_DecimateResult_Get_vertsDeleted( result ), *MR_DecimateResult_Get_facesDeleted( result ) );
    MR_DecimateResult_Destroy( result );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "decimated_mesh.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail;
    }

    rc = EXIT_SUCCESS;
fail:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
