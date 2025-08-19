#include <MRCMesh/MRAffineXf.h>
#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBoolean.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // First, create a unit sphere.
    float radius = 1.f; // Set radius for the test
    int32_t horizontalResolution = 64; // Increase horizontal resolution
    int32_t verticalResolution = 64; // Increase vertical resolution

    MR_Mesh* sphere1 = MR_makeUVSphere( &radius, &horizontalResolution, &verticalResolution );

    // Create a copy of this sphere and offset it.
    MR_Mesh* sphere2 = MR_Mesh_ConstructFromAnother( MR_PassBy_Copy, sphere1 );
    MR_Vector3f xfTranslation = {.x = 0.7f};
    MR_AffineXf3f xf = MR_AffineXf3f_translation( &xfTranslation );
    MR_Mesh_transform( sphere2, &xf, NULL );

    // Perform the boolean operation.
    MR_BooleanResult* result = MR_boolean_4_const_MR_Mesh_ref( sphere1, sphere2, MR_BooleanOperation_Intersection, NULL );
    if ( !MR_BooleanResult_valid( result ) )
    {
        fprintf( stderr, "Failed to perform boolean: %s\n", MR_std_string_Data( MR_BooleanResult_Get_errorString( result ) ) );
        goto fail;
    }

    // Save result to an STL file.
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( MR_BooleanResult_Get_mesh( result ), "out_boolean.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail;
    }

    rc = EXIT_SUCCESS;
fail:
    MR_BooleanResult_Destroy( result );
    MR_Mesh_Destroy( sphere2 );
    MR_Mesh_Destroy( sphere1 );
    return rc;
}
