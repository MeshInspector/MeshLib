#include <MRCMesh/MRICP.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshOrPoints.h>
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

    // Load meshes:

    // First mesh, which will be moved.
    MR_expected_MR_Mesh_std_string* meshFloatingEx = MR_MeshLoad_fromAnySupportedFormat_2( "meshA.stl", NULL, NULL );
    MR_Mesh* meshFloating = MR_expected_MR_Mesh_std_string_GetMutableValue( meshFloatingEx );
    if ( !meshFloating )
    {
        fprintf( stderr, "Failed to load mesh A: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshFloatingEx ) ) );
        goto fail_mesh_loading_a;
    }

    // Second mesh, static.
    MR_expected_MR_Mesh_std_string* meshReferenceEx = MR_MeshLoad_fromAnySupportedFormat_2( "meshB.stl", NULL, NULL );
    MR_Mesh* meshReference = MR_expected_MR_Mesh_std_string_GetMutableValue( meshReferenceEx );
    if ( !meshReference )
    {
        fprintf( stderr, "Failed to load mesh B: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshReferenceEx ) ) );
        goto fail_mesh_loading_b;
    }

    // Prepare ICP parameters
    MR_Box3f bbox = MR_Mesh_computeBoundingBox_1( meshReference, NULL );
    float diagonal = MR_Box3f_diagonal( &bbox );
    float icpSamplingVoxelSize = diagonal * 0.01f; // To sample points from object
    MR_ICPProperties* icpParams = MR_ICPProperties_DefaultConstruct();
    MR_ICPProperties_Set_distThresholdSq( icpParams, diagonal * diagonal * 0.01f ); // Use points pairs with maximum distance specified
    MR_ICPProperties_Set_exitVal( icpParams, diagonal * 0.003f ); // Stop when distance reached

    // Calculate transformation
    MR_MeshOrPoints* fltMop = MR_MeshOrPoints_Construct_MR_Mesh( meshFloating );
    MR_MeshOrPoints* refMop = MR_MeshOrPoints_Construct_MR_Mesh( meshFloating );
    MR_MeshOrPointsXf* flt = MR_MeshOrPointsXf_ConstructFrom( fltMop, MR_AffineXf3f_DefaultConstruct() );
    MR_MeshOrPointsXf* ref = MR_MeshOrPointsXf_ConstructFrom( refMop, MR_AffineXf3f_DefaultConstruct() );
    MR_MeshOrPoints_Destroy( fltMop );
    MR_MeshOrPoints_Destroy( refMop );

    MR_ICP* icp = MR_ICP_Construct_3( flt, ref, icpSamplingVoxelSize );
    MR_ICP_setParams( icp, icpParams );
    MR_ICPProperties_Destroy( icpParams );
    MR_AffineXf3f xf = MR_ICP_calculateTransformation( icp );

    // Transform floating mesh
    MR_Mesh_transform( meshFloating, &xf, NULL );

    // Output information string
    MR_std_string* info = MR_ICP_getStatusInfo( icp );
    printf( "%s\n", MR_std_string_Data( info ) );
    MR_std_string_Destroy( info );

    printf("Final transform:\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n", xf.A.x.x, xf.A.x.y, xf.A.x.z, xf.b.x, xf.A.y.x, xf.A.y.y, xf.A.y.z, xf.b.y, xf.A.z.x, xf.A.z.y, xf.A.z.z, xf.b.z);

    MR_ICP_Destroy( icp );
    MR_MeshOrPointsXf_Destroy( flt );
    MR_MeshOrPointsXf_Destroy( ref );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( meshFloating, "meshA_icp.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;

fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_mesh_loading_b:
    MR_expected_MR_Mesh_std_string_Destroy( meshReferenceEx );
fail_mesh_loading_a:
    MR_expected_MR_Mesh_std_string_Destroy( meshFloatingEx );
    return rc;
}
