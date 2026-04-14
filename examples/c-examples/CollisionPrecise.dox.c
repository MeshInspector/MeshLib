#include "MRCMesh/MRMakeSphereMesh.h"
#include "MRCMesh/MRAffineXf.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshPart.h"
#include "MRCMesh/MRMeshCollidePrecise.h"
#include "MRCMesh/MRPrecisePredicates3.h"
#include "MRCMesh/MRBitSet.h"
#include <MRCMisc/std_vector_MR_VarEdgeTri.h>
#include <MRCMisc/std_string.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    MR_Mesh* meshA = MR_makeUVSphere( NULL, NULL, NULL ); // make mesh A
    MR_Mesh* meshB = MR_makeUVSphere( NULL, NULL, NULL ); // make mesh B
    MR_Vector3f shift = MR_Vector3f_Construct_3( 0.1f, 0.1f, 0.1f );
    MR_AffineXf3f xf = MR_AffineXf3f_translation( &shift );
    MR_Mesh_transform( meshB, &xf, NULL ); // shift mesh B for better demonstration

    MR_MeshPart* mpA = MR_MeshPart_Construct( meshA, NULL );
    MR_MeshPart* mpB = MR_MeshPart_Construct( meshB, NULL );

    // create converters to integer field (needed for absolute precision predicates)
    MR_CoordinateConverters* converters = MR_getVectorConverters_3( mpA, mpB, NULL );
    // find each intersecting edge/triangle pair
    MR_std_vector_MR_VarEdgeTri* collidingFaceEdges = MR_findCollidingEdgeTrisPrecise_5(
        mpA, mpB, MR_PassBy_Copy, MR_CoordinateConverters_GetMutable_toInt( converters ), NULL, NULL );

    // print pairs of edges triangles
    for ( MR_uint64_t i = 0; i < MR_std_vector_MR_VarEdgeTri_size( collidingFaceEdges ); ++i )
    {
        MR_VarEdgeTri* edgeTri = MR_std_vector_MR_VarEdgeTri_at_mut( collidingFaceEdges, i );
        bool edgeATriB = MR_VarEdgeTri_isEdgeATriB( edgeTri );
        MR_FaceId tri = MR_VarEdgeTri_tri( edgeTri );
        if ( edgeATriB )
            fprintf( stdout, "edgeA: %d, triB: %d\n", *MR_EdgeId_get( MR_VarEdgeTri_GetMutable_edge( edgeTri ) ), *MR_FaceId_get( &tri ) );
        else
            fprintf( stdout, "triA: %d, edgeB: %d\n", *MR_FaceId_get( &tri ), *MR_EdgeId_get( MR_VarEdgeTri_GetMutable_edge( edgeTri ) ) );
    }

    MR_std_vector_MR_VarEdgeTri_Destroy( collidingFaceEdges );
    MR_CoordinateConverters_Destroy( converters );

    MR_MeshPart_Destroy( mpB );
    MR_MeshPart_Destroy( mpA );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );

    rc = EXIT_SUCCESS;
    return rc;
}
