#include <MRCMesh/MRPointsLoad.h>
#include <MRCMesh/MRPointCloud.h>
#include <MRCMesh/MRPointCloudTriangulation.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_MR_PointCloud_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_optional_MR_Mesh.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;
    // load points
    MR_expected_MR_PointCloud_std_string* loadRes = MR_PointsLoad_fromAnySupportedFormat_2( "Points.ply", NULL, NULL );
    MR_PointCloud* pointCloud = MR_expected_MR_PointCloud_std_string_value_mut( loadRes );
    if ( !pointCloud )
    {
        fprintf( stderr, "Failed to load points: %s\n", MR_std_string_data( MR_expected_MR_PointCloud_std_string_error( loadRes ) ) );
        goto fail_load; // error while loading file
    }
    MR_std_optional_MR_Mesh* triangulationRes = MR_triangulatePointCloud( pointCloud, NULL, NULL );
    MR_Mesh* mesh = MR_std_optional_MR_Mesh_value_mut( triangulationRes );
    if ( !mesh )
    {
        fprintf( stderr, "Triangulation canceled" );
        goto fail_triangulation; // can be nullopt only if canceled by progress callback
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "mesh.ply", NULL, NULL );
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_triangulation:
    MR_std_optional_MR_Mesh_Destroy( triangulationRes );
fail_load:
    MR_expected_MR_PointCloud_std_string_Destroy( loadRes );
    return rc;
}
