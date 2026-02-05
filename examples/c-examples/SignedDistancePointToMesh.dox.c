#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshProject.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/std_optional_MR_SignedDistanceToMeshResult.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshRes = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.ctm", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_value_mut( meshRes );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( meshRes ) ) );
        goto fail_load;
    }

    MR_Vector3f point = MR_Vector3f_Construct_3( 1.5f, 1.0f, 0.5f );
    MR_MeshPart* mp = MR_MeshPart_Construct( mesh, NULL );

    MR_std_optional_MR_SignedDistanceToMeshResult* resOpt = MR_findSignedDistance_MR_Vector3f( &point, mp, NULL, NULL );
    MR_SignedDistanceToMeshResult* res = MR_std_optional_MR_SignedDistanceToMeshResult_value_mut( resOpt );
    if ( res )
        fprintf( stdout, "Signed distance from point to mesh: %f\n", *MR_SignedDistanceToMeshResult_Get_dist( res ) );

    rc = EXIT_SUCCESS;

    MR_std_optional_MR_SignedDistanceToMeshResult_Destroy( resOpt );
    MR_MeshPart_Destroy( mp );
fail_load:
    MR_expected_MR_Mesh_std_string_Destroy( meshRes );
    return rc;
}
