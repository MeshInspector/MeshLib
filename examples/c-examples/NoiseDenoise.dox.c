#include <MRCMesh/MRAddNoise.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRNormalDenoising.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail_load;
    }

    // Add noise to the mesh
    MR_Box3f box = MR_Mesh_computeBoundingBox_1( mesh, NULL );
    MR_NoiseSettings* noiseSettings = MR_NoiseSettings_DefaultConstruct();
    MR_NoiseSettings_Set_sigma( noiseSettings, MR_Box3f_diagonal( &box ) * 0.0001f );
    MR_expected_void_std_string* noiseEx = MR_addNoise_MR_Mesh( mesh, NULL, noiseSettings );
    MR_NoiseSettings_Destroy( noiseSettings );

    if ( !noiseEx )
    {
        fprintf( stderr, "Failed to add noise: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( noiseEx ) ) );
        goto fail_noise;
    }

    // Invalidate the mesh because of the external vertex changes
    MR_Mesh_invalidateCaches( mesh, NULL );

    // Save the noised mesh
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "mesh_noised.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    // Denoise the mesh with sharpening for sharp edges
    // see the article "Mesh Denoising via a Novel Mumford-Shah Framework"
    MR_meshDenoiseViaNormals( mesh, NULL );

    // Save the denoised mesh
    MR_expected_void_std_string* saveEx2 = MR_MeshSave_toAnySupportedFormat_3( mesh, "mesh_denoised.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx2 ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx2 ) ) );
        goto fail_save2;
    }

    rc = EXIT_SUCCESS;
fail_save2:
    MR_expected_void_std_string_Destroy( saveEx2 );
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_noise:
    MR_expected_void_std_string_Destroy( noiseEx );
fail_load:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
