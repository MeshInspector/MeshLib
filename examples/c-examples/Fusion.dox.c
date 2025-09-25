#include <MRCMesh/MRPointsLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRPointCloudRadius.h>
#include <MRCMesh/MRPointCloud.h>
#include <MRCMesh/MRBox.h>
#include <MRCVoxels/MRPointsToMeshFusion.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_MR_PointCloud_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // load points
    MR_expected_MR_PointCloud_std_string* loadRes = MR_PointsLoad_fromAnySupportedFormat_2( "Points.ply", NULL, NULL );
    MR_PointCloud* pc = MR_expected_MR_PointCloud_std_string_GetMutableValue( loadRes );
    if ( !pc )
    {
        fprintf( stderr, "Failed to load points: %s\n", MR_std_string_Data( MR_expected_MR_PointCloud_std_string_GetError( loadRes ) ) );
        goto fail_load;
    }

    MR_PointsToMeshParameters* params = MR_PointsToMeshParameters_DefaultConstruct();
    MR_Box3f box = MR_PointCloud_computeBoundingBox_1( pc, NULL );
    float voxelSize = MR_Box3f_diagonal( &box ) * 1e-2f;
    MR_PointsToMeshParameters_Set_voxelSize( params, voxelSize );
    float sigma = max( voxelSize, MR_findAvgPointsRadius( pc, 50, NULL ) );
    MR_PointsToMeshParameters_Set_sigma( params, sigma );
    MR_PointsToMeshParameters_Set_minWeight( params, 1.0f );

    MR_expected_MR_Mesh_std_string* fusionRes = MR_pointsToMeshFusion( pc, params );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( fusionRes );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to fuse points: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( fusionRes ) ) );
        goto fail_fuse;
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "Mesh.ctm", NULL, NULL );
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_fuse:
    MR_expected_MR_Mesh_std_string_Destroy( fusionRes );
    MR_PointsToMeshParameters_Destroy( params );
fail_load:
    MR_expected_MR_PointCloud_std_string_Destroy( loadRes );
    return rc;
}
