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

    // Load meshes:

    // First mesh, which will be moved.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "meshA.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh A: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        goto fail_mesh_loading_a;
    }

    // Second mesh, static.
    MR_expected_MR_Mesh_std_string* mesh2Ex = MR_MeshLoad_fromAnySupportedFormat_2( "meshB.stl", NULL, NULL );
    MR_Mesh* mesh2 = MR_expected_MR_Mesh_std_string_GetMutableValue( mesh2Ex );
    if ( !mesh2 )
    {
        fprintf( stderr, "Failed to load mesh B: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( mesh2Ex ) ) );
        goto fail_mesh_loading_b;
    }

    // Unite meshes
    MR_Mesh_addMesh_3( mesh, mesh2, NULL, NULL );

    // Find holes (expect that there are exactly 2 holes)
    MR_std_vector_MR_EdgeId* edges = MR_MeshTopology_findHoleRepresentiveEdges( MR_Mesh_Get_topology( mesh ), NULL );
    if ( MR_std_vector_MR_EdgeId_Size( edges ) != 2 )
    {
        fprintf( stderr, "Expected exactly 2 holes to stitch, but found %zu\n", MR_std_vector_MR_EdgeId_Size( edges ) );
        goto fail_not_two_holes;
    }

    // Connect two holes
    MR_StitchHolesParams* params = MR_StitchHolesParams_DefaultConstruct();

    MR_FillHoleMetric* metric = MR_getUniversalMetric( mesh );
    MR_StitchHolesParams_Set_metric( params, MR_PassBy_Move, metric );
    MR_FillHoleMetric_Destroy( metric );

    // We also have a version of this function (`MR_buildCylinderBetweenTwoHoles_2()`) that finds the two holes automatically.
    // Here we've found them manually for demonstration purposes.
    MR_buildCylinderBetweenTwoHoles_4( mesh, *MR_std_vector_MR_EdgeId_At( edges, 0 ), *MR_std_vector_MR_EdgeId_At( edges, 1 ), params );
    MR_StitchHolesParams_Destroy( params );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "MeshStitched.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
    {
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
        goto fail_save;
    }

    rc = EXIT_SUCCESS;
fail_save:
    MR_expected_void_std_string_Destroy( saveEx );
fail_not_two_holes:
    MR_std_vector_MR_EdgeId_Destroy( edges );
fail_mesh_loading_b:
    MR_expected_MR_Mesh_std_string_Destroy( mesh2Ex );
fail_mesh_loading_a:
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
