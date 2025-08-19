#include "TestMacros.h"

#include "MRPointsToMeshProjector.h"

#include "MRCMesh/MRAffineXf.h"
#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRPointsToMeshProjector.h"
#include "MRCMesh/MRVector.h"

void testFindSignedDistances( void )
{
    MR_Mesh* meshA = createCube();
    MR_Mesh* meshB = createCube();

    const MR_Vector3f shift = { 1.f, 2.f, 3.f };
    const MR_AffineXf3f xf = MR_AffineXf3f_translation( &shift );

    MR_MeshProjectionParameters* params = MR_MeshProjectionParameters_DefaultConstruct();
    MR_MeshProjectionParameters_Set_xf( params, &xf );

    MR_VertScalars* results = MR_findSignedDistances_4( meshA, meshB, params, NULL );
    MR_MeshProjectionParameters_Destroy( params );

    TEST_ASSERT_INT_EQUAL( (int)MR_VertScalars_size( results ), (int)MR_VertCoords_size( MR_Mesh_Get_points( meshB ) ) );

    float maxDist = 0.f;
    for ( int i = 0; i < MR_VertScalars_size( results ); i++ )
        if ( maxDist < fabsf( *MR_VertScalars_index_const( results, (MR_VertId){i} ) ) )
            maxDist = fabsf( *MR_VertScalars_index_const( results, (MR_VertId){i} ) );
    TEST_ASSERT_FLOAT_EQUAL_APPROX( maxDist, MR_Vector3f_length( &shift ), 1e-6f )

    MR_VertScalars_Destroy( results );

    MR_Mesh_Destroy( meshB );
    MR_Mesh_Destroy( meshA );
}
