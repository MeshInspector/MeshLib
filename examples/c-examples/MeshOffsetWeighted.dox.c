#include <MRCMesh/MRTorus.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRVector.h>
#include <MRCVoxels/MRWeightedPointsShell.h>
#include <MRCMesh/MRClosestWeightedPoint.h>
#include <MRCVoxels/MROffset.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/std_vector_MR_Vector3f.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define abs(a) a >= 0 ? a : -a;

int main( void )
{
    int rc = EXIT_FAILURE;
    // Create some mesh
    MR_Mesh* mesh = MR_makeTorus( NULL, NULL, NULL, NULL, NULL );

    // Create VertScalars obj with weights for every vertex
    MR_uint64_t vertSize = MR_MeshTopology_vertSize( MR_Mesh_Get_topology( mesh ) );
    MR_VertScalars* scalars = MR_VertScalars_Construct_1_uint64_t( vertSize );
    float maxWeight = 0.0f;
    for ( MR_uint64_t i = 0; i < vertSize; ++i )
    {
        const MR_std_vector_MR_Vector3f* points = MR_VertCoords_Get_vec_( MR_Mesh_Get_points( mesh ) );
        float weight = abs( MR_std_vector_MR_Vector3f_At( points, i )->x / 5.0f );
        MR_VertScalars_data( scalars )[i] = weight;
        if ( weight > maxWeight )
        {
            maxWeight = weight;
        }
    }

    MR_WeightedShell_ParametersMetric* params = MR_WeightedShell_ParametersMetric_DefaultConstruct();
    // Algorithm is voxel based, voxel size affects performance and form of result mesh
    MR_WeightedShell_ParametersBase* base = MR_WeightedShell_ParametersMetric_MutableUpcastTo_MR_WeightedShell_ParametersBase( params );
    MR_MeshPart* mp = MR_MeshPart_Construct( mesh, NULL );
    MR_WeightedShell_ParametersBase_Set_voxelSize( base, MR_suggestVoxelSize( mp, 1000.0f ) );
    // common basic offset applied for all point
    // Vertex-specific weighted offsets applied after the basic one
    MR_WeightedShell_ParametersBase_Set_offset( base, 0.2f );
    MR_DistanceFromWeightedPointsParams* dist = MR_WeightedShell_ParametersMetric_GetMutable_dist( params );
    MR_DistanceFromWeightedPointsParams_Set_maxWeight( dist, maxWeight );

    MR_expected_MR_Mesh_std_string* res = MR_WeightedShell_meshShell_3_MR_VertScalars( mesh, scalars, params );
    MR_Mesh* resMesh = MR_expected_MR_Mesh_std_string_GetMutableValue( res );
    if ( !resMesh )
    {
        fprintf( stderr, "Failed to create shell: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( res ) ) );
        goto fail_shell;
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( resMesh, "offset_weighted.ctm", NULL, NULL );
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save; // error while saving file
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );

fail_shell:
    MR_expected_MR_Mesh_std_string_Destroy( res );

    MR_MeshPart_Destroy( mp );
    MR_WeightedShell_ParametersMetric_Destroy( params );
    MR_VertScalars_Destroy( scalars );
    MR_Mesh_Destroy( mesh );
    return rc;
}
