#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshMeshDistance.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* mesh1Res = MR_MeshLoad_fromAnySupportedFormat_2( "mesh1.ctm", NULL, NULL );
    MR_Mesh* mesh1 = MR_expected_MR_Mesh_std_string_value_mut( mesh1Res );
    if ( !mesh1 )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( mesh1Res ) ) );
        goto fail_load_1;
    }

    MR_expected_MR_Mesh_std_string* mesh2Res = MR_MeshLoad_fromAnySupportedFormat_2( "mesh2.ctm", NULL, NULL );
    MR_Mesh* mesh2 = MR_expected_MR_Mesh_std_string_value_mut( mesh2Res );
    if ( !mesh2 )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( mesh2Res ) ) );
        goto fail_load_2;
    }

    MR_MeshPart* mp1 = MR_MeshPart_Construct( mesh1, NULL );
    MR_MeshPart* mp2 = MR_MeshPart_Construct( mesh2, NULL );

    MR_MeshMeshSignedDistanceResult* res = MR_findSignedDistance_MR_MeshPart( mp1, mp2, NULL, NULL );
    const float* dist = MR_MeshMeshSignedDistanceResult_Get_signedDist( res );

    fprintf( stdout, "Signed distance between meshes is: %f\n", *dist );

    MR_MeshMeshSignedDistanceResult_Destroy( res );
    MR_MeshPart_Destroy( mp2 );
    MR_MeshPart_Destroy( mp1 );

    rc = EXIT_SUCCESS;

fail_load_2:
    MR_expected_MR_Mesh_std_string_Destroy( mesh2Res );
fail_load_1:
    MR_expected_MR_Mesh_std_string_Destroy( mesh1Res );
    return rc;
}
