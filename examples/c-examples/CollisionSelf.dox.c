#include "MRCMesh/MRTorus.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRMeshPart.h"
#include "MRCMesh/MRMeshCollide.h"
#include "MRCMesh/MRBitSet.h"
#include "MRCMesh/MRFaceFace.h"
#include <MRCMisc/std_string.h>
#include <MRCMisc/expected_std_vector_MR_FaceFace_std_string.h>
#include <MRCMisc/expected_MR_FaceBitSet_std_string.h>
#include <MRCMisc/expected_bool_std_string.h>
#include <MRCMisc/std_vector_MR_FaceFace.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main( void )
{
    int rc = EXIT_FAILURE;

    // make torus with self-intersections
    MR_Mesh* mesh = MR_makeTorusWithSelfIntersections( NULL, NULL, NULL, NULL, NULL );
    MR_MeshPart* mp = MR_MeshPart_Construct( mesh, NULL );

    // find self-intersecting faces pairs
    MR_expected_std_vector_MR_FaceFace_std_string* selfCollidingPairsRes = MR_findSelfCollidingTriangles_4( mp, MR_PassBy_DefaultArgument, NULL, NULL, NULL );
    MR_std_vector_MR_FaceFace* selfCollidingPairs = MR_expected_std_vector_MR_FaceFace_std_string_value_mut( selfCollidingPairsRes );
    if ( !selfCollidingPairs )
    {
        // check error
        fprintf( stderr, "%s\n", MR_std_string_data( MR_expected_std_vector_MR_FaceFace_std_string_error( selfCollidingPairsRes ) ) );
        goto fail_self_collision_pairs;
    }

    for ( MR_uint64_t i = 0; i < MR_std_vector_MR_FaceFace_size( selfCollidingPairs ); ++i )
    {
        const MR_FaceFace* ff = MR_std_vector_MR_FaceFace_at( selfCollidingPairs, i );
        // print each pair
        fprintf( stdout, "%d %d\n", MR_FaceFace_Get_aFace( ff )->id_, MR_FaceFace_Get_bFace( ff )->id_ );
    }

    // find bitset of self-intersecting faces
    MR_expected_MR_FaceBitSet_std_string* selfCollidingBitSetRes = MR_findSelfCollidingTrianglesBS( mp, MR_PassBy_DefaultArgument, NULL, NULL, NULL );
    MR_FaceBitSet* selfCollidingBitSet = MR_expected_MR_FaceBitSet_std_string_value_mut( selfCollidingBitSetRes );
    if ( !selfCollidingBitSet )
    {
        // check error
        fprintf( stderr, "%s\n", MR_std_string_data( MR_expected_MR_FaceBitSet_std_string_error( selfCollidingBitSetRes ) ) );
        goto fail_self_collision_bs;
    }
    // print number of self-intersecting faces
    fprintf( stdout, "%" PRIu64 "\n", MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( selfCollidingBitSet ) ) );

    // fast check if mesh has self-intersections
    MR_expected_bool_std_string* isSelfCollidingRes = MR_findSelfCollidingTriangles_5( mp, NULL, MR_PassBy_DefaultArgument, NULL, NULL, NULL );
    bool* isSelfColliding = MR_expected_bool_std_string_value_mut( isSelfCollidingRes );
    if ( !isSelfColliding )
    {
        // check error
        fprintf( stderr, "%s\n", MR_std_string_data( MR_expected_bool_std_string_error( isSelfCollidingRes ) ) );
        goto fail_self_collision_fast;
    }
    // print number of self-intersecting faces
    fprintf( stdout, "%d\n", *isSelfColliding );

    rc = EXIT_SUCCESS;
fail_self_collision_fast:
    MR_expected_bool_std_string_Destroy( isSelfCollidingRes );
fail_self_collision_bs:
    MR_expected_MR_FaceBitSet_std_string_Destroy( selfCollidingBitSetRes );
fail_self_collision_pairs:
    MR_expected_std_vector_MR_FaceFace_std_string_Destroy( selfCollidingPairsRes );
    MR_MeshPart_Destroy( mp );
    MR_Mesh_Destroy( mesh );
    return rc;
}
