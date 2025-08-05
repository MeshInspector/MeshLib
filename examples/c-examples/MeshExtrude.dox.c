#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshExtrude.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRRegionBoundary.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_void_std_string.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // Load mesh.
    MR_expected_MR_Mesh_std_string* meshEx = MR_MeshLoad_fromAnySupportedFormat_2( "mesh.stl", NULL, NULL );
    MR_Mesh* mesh = MR_expected_MR_Mesh_std_string_GetMutableValue( meshEx );

    // Handle failure to load mesh.
    if ( !mesh )
    {
        fprintf( stderr, "Failed to load mesh: %s\n", MR_std_string_Data( MR_expected_MR_Mesh_std_string_GetError( meshEx ) ) );
        MR_expected_MR_Mesh_std_string_Destroy( meshEx );
        return 1;
    }

    // Select faces to extrude
    MR_FaceBitSet* facesToExtrude = MR_FaceBitSet_DefaultConstruct();
    MR_BitSet_autoResizeSet_2( MR_FaceBitSet_MutableUpcastTo_MR_BitSet( facesToExtrude ), 1, NULL );
    MR_BitSet_autoResizeSet_2( MR_FaceBitSet_MutableUpcastTo_MR_BitSet( facesToExtrude ), 2, NULL );

    // Create duplicated verts on region boundary
    MR_makeDegenerateBandAroundRegion( mesh, facesToExtrude, NULL );

    // Find vertices that will be moved
    MR_VertBitSet* vertsToMove = MR_getIncidentVerts_2_MR_FaceBitSet( MR_Mesh_Get_topology( mesh ), facesToExtrude );
    MR_Vector3f* points = MR_VertCoords_data( MR_Mesh_GetMutable_points( mesh ) );
    MR_Vector3f shift = MR_Vector3f_plusZ();
    size_t numPoints = MR_VertCoords_size( MR_Mesh_GetMutable_points( mesh ) );
    for ( size_t i = 0; i < numPoints; ++i )
        if ( MR_VertBitSet_test( vertsToMove, (MR_VertId){ i } ) )
            points[i] = MR_add_MR_Vector3f( &points[i], &shift );

    // Invalidate internal caches after manual changing
    MR_Mesh_invalidateCaches( mesh, NULL );

    // Save result
    MR_expected_void_std_string* saveEx = MR_MeshSave_toAnySupportedFormat_3( mesh, "extruded_mesh.stl", NULL, NULL);
    if ( MR_expected_void_std_string_GetError( saveEx ) )
        fprintf( stderr, "Failed to save mesh: %s\n", MR_std_string_Data( MR_expected_void_std_string_GetError( saveEx ) ) );
    else
        rc = EXIT_SUCCESS;

    MR_VertBitSet_Destroy( vertsToMove );
    MR_FaceBitSet_Destroy( facesToExtrude );
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return rc;
}
