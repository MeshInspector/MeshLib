#include "MRCMesh/MRMakeSphereMesh.h"
#include "MRCMesh/MRAffineXf.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshPart.h"
#include "MRCMesh/MRMeshCollide.h"
#include "MRCMesh/MRFaceFace.h"
#include "MRCMesh/MRBitSet.h"
#include <MRCMisc/std_vector_MR_FaceFace.h>
#include <MRCMisc/std_pair_MR_FaceBitSet_MR_FaceBitSet.h>
#include <MRCMisc/std_string.h>

#include <inttypes.h>
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

    MR_std_vector_MR_FaceFace* collidingFacePairs = MR_findCollidingTriangles( mpA, mpB, NULL, NULL ); // find each pair of colliding faces

    for ( MR_uint64_t i = 0; i < MR_std_vector_MR_FaceFace_size( collidingFacePairs ); ++i )
    {
        // print each pair of colliding faces
        const MR_FaceFace* ff = MR_std_vector_MR_FaceFace_at( collidingFacePairs, i );
        fprintf( stdout, "%d %d\n", MR_FaceFace_Get_aFace( ff )->id_, MR_FaceFace_Get_bFace( ff )->id_ );
    }

    // find bitsets of colliding faces
    MR_std_pair_MR_FaceBitSet_MR_FaceBitSet* collidingBitSets = MR_findCollidingTriangleBitsets( mpA, mpB, NULL );

    MR_uint64_t numColA = MR_BitSet_count(
        MR_FaceBitSet_UpcastTo_MR_BitSet(
            MR_std_pair_MR_FaceBitSet_MR_FaceBitSet_first( collidingBitSets ) ) );
    MR_uint64_t numColB = MR_BitSet_count(
        MR_FaceBitSet_UpcastTo_MR_BitSet(
            MR_std_pair_MR_FaceBitSet_MR_FaceBitSet_second( collidingBitSets ) ) );
    fprintf( stdout, "%" PRIu64 "\n", numColA ); // print number of colliding faces from mesh A
    fprintf( stdout, "%" PRIu64 "\n", numColB ); // print number of colliding faces from mesh B


    // fast check if mesh A and mesh B collide
    bool firstIntersectionsOnly = true;
    MR_std_vector_MR_FaceFace* fastCheck = MR_findCollidingTriangles( mpA, mpB, NULL, &firstIntersectionsOnly );
    bool isColliding = !MR_std_vector_MR_FaceFace_empty( fastCheck );
    if ( isColliding )
        fprintf( stdout, "true\n" );
    else
        fprintf( stdout, "false\n" );

    MR_std_vector_MR_FaceFace_Destroy( fastCheck );
    MR_std_pair_MR_FaceBitSet_MR_FaceBitSet_Destroy( collidingBitSets );
    MR_std_vector_MR_FaceFace_Destroy( collidingFacePairs );

    MR_MeshPart_Destroy( mpB );
    MR_MeshPart_Destroy( mpA );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );

    rc = EXIT_SUCCESS;
    return rc;
}
