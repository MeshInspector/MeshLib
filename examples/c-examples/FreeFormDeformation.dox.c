#include <MRCMesh/MRFreeFormDeformer.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    // Load mesh
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    // Handle failure to load mesh
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        MR_expected_MR_Mesh_std_string_Destroy( meshEx );
        return 1;
    }

    // Compute mesh bounding box
    MR_Box3f box = MR_Mesh_computeBoundingBox_1( mesh, NULL );

    // Construct deformer on mesh vertices
    MR_FreeFormDeformer* ffDeformer = MR_FreeFormDeformer_Construct_MR_Mesh( mesh, NULL );

    // Init deformer with 3x3 grid on mesh box
    MR_Vector3i resolution = MR_Vector3i_diagonal( 3 );
    MR_FreeFormDeformer_init( ffDeformer, &resolution, &box );

    // Move some control points of the grid to the center
    MR_Vector3i controlPoints[] = {
        { 1, 1, 0 },
        { 1, 1, 2 },
        { 0, 1, 1 },
        { 2, 1, 1 },
        { 1, 0, 1 },
        { 1, 2, 1 },
    };
    MR_Vector3f center = MR_Box3f_center( &box );
    for ( int i = 0; i < 6; ++i )
        MR_FreeFormDeformer_setRefGridPointPosition( ffDeformer, &controlPoints[i], &center );

    // Apply the deformation to the mesh vertices
    MR_FreeFormDeformer_apply( ffDeformer );

    // Invalidate the mesh because of external vertex changes
    MR_Mesh_invalidateCaches( mesh, NULL );

    // Save deformed mesh
    MR_MeshSave_toAnySupportedFormat_3( mesh, "deformed_mesh.stl", NULL, NULL );

    MR_FreeFormDeformer_Destroy( ffDeformer );
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return EXIT_SUCCESS;
}
