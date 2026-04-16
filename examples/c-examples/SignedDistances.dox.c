#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRPointsToMeshProjector.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* mesh1Res = MR_MeshLoad_fromAnySupportedFormat_2( "mesh1.ctm", NULL, NULL );
    MR_Mesh* refMesh = MR_expected_MR_Mesh_std_string_value_mut( mesh1Res );
    if ( !refMesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( mesh1Res ) ) );
        goto fail_load_1;
    }

    MR_expected_MR_Mesh_std_string* mesh2Res = MR_MeshLoad_fromAnySupportedFormat_2( "mesh2.ctm", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_value_mut( mesh2Res );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( mesh2Res ) ) );
        goto fail_load_2;
    }

    // get object of VertScalars - set of distances between points of target mesh and reference mesh
    MR_VertScalars* vertDistances = MR_findSignedDistances_4( refMesh, mesh, NULL, NULL );
    float minDist = FLT_MAX;
    float maxDist = -FLT_MAX;
    for ( MR_uint64_t i = 0; i < MR_VertScalars_size( vertDistances ); ++i )
    {
        float dist = MR_VertScalars_data( vertDistances )[i];
        if ( dist < minDist )
            minDist = dist;
        if ( dist > maxDist )
            maxDist = dist;
    }

    fprintf( stdout, "Distance between reference mesh and the closest point of target mesh is: %f\n", minDist );
    fprintf( stdout, "Distance between reference mesh and the farthest point of target mesh is: %f\n", maxDist );


    rc = EXIT_SUCCESS;

    MR_VertScalars_Destroy( vertDistances );
fail_load_2:
    MR_expected_MR_Mesh_std_string_Destroy( mesh2Res );
fail_load_1:
    MR_expected_MR_Mesh_std_string_Destroy( mesh1Res );
    return rc;
}
