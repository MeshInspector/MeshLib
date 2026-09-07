#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshFillHole.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshMetrics.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>
#include <MRCMisc/std_vector_MR_EdgeId.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_value_mut( meshEx );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_data( MR_expected_MR_Mesh_std_string_error( meshEx ) ) );
        goto fail_mesh_loading;
    }

    // Find single edge for each hole in mesh
    MR_std_vector_MR_EdgeId* holeEdges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );

    // Setup filling parameters
    MR_FillHoleParams* params = MR_FillHoleParams_DefaultConstruct();
    MR_FillHoleMetric* metric = MR_getUniversalMetric( mesh );
    MR_FillHoleParams_Set_metric( params, MR_PassBy_Move, metric );
    MR_FillHoleMetric_Destroy( metric );

    // Alternatively, MR_fillHoles( mesh, holeEdges, params ) fills all holes at once.
    for ( size_t i = 0; i < MR_std_vector_MR_EdgeId_size( holeEdges ); ++i )
    {
        // Fill hole represented by `e`
        MR_EdgeId e = *MR_std_vector_MR_EdgeId_at( holeEdges, i );
        MR_fillHole( mesh, e, params );
    }
    MR_FillHoleParams_Destroy( params );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "filledMesh.stl", NULL, NULL);
    if ( MR_expected_void_std_string_error( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_data( MR_expected_void_std_string_error( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
    MR_std_vector_MR_EdgeId_Destroy( holeEdges );
fail_mesh_loading:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
