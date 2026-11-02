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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;
    // Create some mesh
    MR_Mesh* mesh = MR_makeTorus( NULL, NULL, NULL, NULL, NULL );

    // Create VertScalars obj with weights for every vertex
    size_t vertSize = MR_MeshTopology_vertSize( MR_Mesh_Get_topology( mesh ) );
    MR_VertScalars* scalars = MR_VertScalars_Construct( vertSize );
    float maxWeight = 0.0f;
    for ( size_t i = 0; i < vertSize; ++i )
    {
        const MR_Vector3f* point = MR_VertCoords_index( MR_Mesh_Get_points( mesh ), (MR_VertId){ (int)i } );
        float weight = fabsf( point->x / 5.0f ); // Individual extra offset sizes for points
        MR_VertScalars_data_mut( scalars )[i] = weight;
        maxWeight = fmaxf( maxWeight, weight ); // should always have maximum between weights provided
    }

    MR_WeightedShell_ParametersMetric* params = MR_WeightedShell_ParametersMetric_DefaultConstruct();
    // Algorithm is voxel based, voxel size affects performance and form of result mesh
    MR_MeshPart* mp = MR_MeshPart_Construct( mesh, NULL );
    MR_WeightedShell_ParametersMetric_Set_voxelSize( params, MR_suggestVoxelSize( mp, 1000.0f ) );
    // common basic offset applied for all point
    // Vertex-specific weighted offsets applied after the basic one
    MR_WeightedShell_ParametersMetric_Set_offset( params, 0.2f );
    MR_DistanceFromWeightedPointsParams* dist = MR_WeightedShell_ParametersMetric_GetMutable_dist( params );
    MR_DistanceFromWeightedPointsParams_Set_maxWeight( dist, maxWeight );

    MR_expected_MR_Mesh_std_string* res = MR_WeightedShell_meshShell( mesh, scalars, params );
    MR_Mesh* resMesh = MR_expected_MR_Mesh_std_string_value_mut( res );
    if ( !resMesh )
    {
        fprintf( stderr, "Failed to create shell: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( res ) ) );
        goto fail_shell;
    }

    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat( resMesh, "offset_weighted.ctm", NULL, NULL );
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
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
