#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRExpandShrink.h>
#include <MRCMesh/MRLaplacian.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshLoad.h>
#include <MRCMesh/MRMeshSave.h>
#include <MRCMesh/MRMeshTopology.h>
#include <MRCMesh/MRVector.h>
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

    // Construct deformer on the mesh vertices
    MR_Laplacian* lDeformer = MR_Laplacian_Construct_1( mesh );

    // Find an area for the deformation anchor points
    const MR_VertBitSet* verts = MR_MeshTopology_getValidVerts( MR_Mesh_Get_topology( mesh ) );
    MR_VertId ancV0 = MR_VertBitSet_find_first( verts );
    MR_VertId ancV1 = MR_VertBitSet_find_last( verts );
    // Mark the anchor points in the free area
    MR_VertBitSet* freeVerts = MR_VertBitSet_DefaultConstruct();
    MR_BitSet_resize( MR_VertBitSet_MutableUpcastTo_MR_BitSet( freeVerts ), MR_BitSet_size( MR_VertBitSet_UpcastTo_MR_BitSet( verts ) ), NULL );

    MR_BitSet_set_2( MR_VertBitSet_MutableUpcastTo_MR_BitSet( freeVerts ), ancV0.id_, true );
    MR_BitSet_set_2( MR_VertBitSet_MutableUpcastTo_MR_BitSet( freeVerts ), ancV1.id_, true );
    // Expand the free area
    MR_expand_MR_VertBitSet( MR_Mesh_Get_topology( mesh ), freeVerts, &(int32_t){5} );

    // Initialize laplacian
    MR_Laplacian_init( lDeformer, freeVerts, MR_EdgeWeights_Cotan, &(MR_VertexMass){MR_VertexMass_NeiArea}, NULL );

    MR_Box3f bbox = MR_Mesh_computeBoundingBox_1( mesh, NULL );
    float shiftAmount = MR_Box3f_diagonal( &bbox ) * 0.01f;
    // Fix the anchor vertices in the required position
    const MR_VertCoords* points = MR_Mesh_Get_points( mesh );
    MR_Vector3f posV0 = MR_Mesh_normal_MR_VertId( mesh, ancV0 );
    posV0 = MR_mul_MR_Vector3f_float( &posV0, shiftAmount );
    posV0 = MR_add_MR_Vector3f( MR_VertCoords_index_const( points, ancV0 ), &posV0 );
    MR_Laplacian_fixVertex_3( lDeformer, ancV0, &posV0, NULL );
    MR_Vector3f posV1 = MR_Mesh_normal_MR_VertId( mesh, ancV1 );
    posV1 = MR_mul_MR_Vector3f_float( &posV1, shiftAmount );
    posV1 = MR_add_MR_Vector3f( MR_VertCoords_index_const( points, ancV1 ), &posV1 );
    MR_Laplacian_fixVertex_3( lDeformer, ancV1, &posV1, NULL );

    // Move the free vertices according to the anchor ones
    MR_Laplacian_apply( lDeformer );

    // Invalidate the mesh because of the external vertex changes
    MR_Mesh_invalidateCaches( mesh, NULL );

    // Save the deformed mesh
    MR_MeshSave_toAnySupportedFormat_3( mesh, "deformed_mesh.stl", NULL, NULL );

    MR_VertBitSet_Destroy( freeVerts );
    MR_Laplacian_Destroy( lDeformer );
    MR_expected_MR_Mesh_std_string_Destroy( meshEx );
    return EXIT_SUCCESS;
}
