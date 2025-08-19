#include "TestMacros.h"
#include "MRExpandShrink.h"

#include "MRCMesh/MRMesh.h"
#include "MRCMesh/MRBitSet.h"
#include "MRCMesh/MRExpandShrink.h"
#include "MRCMesh/MRMakeSphereMesh.h"

void testExpandShrink( void )
{
    MR_SphereParams *params = MR_SphereParams_DefaultConstruct();
    MR_SphereParams_Set_numMeshVertices( params, 3000 );
    MR_SphereParams_Set_radius( params, 1.0f );

    MR_Mesh* mesh = MR_makeSphere( params );
    MR_SphereParams_Destroy( params );

    const MR_MeshTopology* top = MR_Mesh_Get_topology( mesh );

    MR_FaceId face; face.id_ = 0;
    MR_FaceBitSet* region = MR_expand_MR_FaceId( top, face, 3 );

    int num = (int)MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT_INT_EQUAL( num, 75 );

    MR_expand_MR_FaceBitSet( top, region, &(int){3} );
    num = (int)MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT( num == 274 || // without FMA instruction (default settings for x86 or old compilers for ARM)
                 num == 284 ); // with FMA instruction (modern compilers for ARM)

    MR_shrink_MR_FaceBitSet( top, region, &(int){3} );
    num = (int)MR_BitSet_count( MR_FaceBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT_INT_EQUAL( num, 75 );

    MR_FaceBitSet_Destroy( region );
    MR_Mesh_Destroy( mesh );
}

void testExpandShrinkVerts( void )
{
    MR_SphereParams *params = MR_SphereParams_DefaultConstruct();
    MR_SphereParams_Set_numMeshVertices( params, 3000 );
    MR_SphereParams_Set_radius( params, 1.0f );

    MR_Mesh* mesh = MR_makeSphere( params );
    MR_SphereParams_Destroy( params );

    const MR_MeshTopology* top = MR_Mesh_Get_topology( mesh );

    MR_VertId vert; vert.id_ = 0;
    MR_VertBitSet* region = MR_expand_MR_VertId( top, vert, 3 );

    size_t num = MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT( num == 37 );

    MR_expand_MR_VertBitSet( top, region, &(int){3} );
    num = MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT( num > 37 ); //platform dependent results

    MR_shrink_MR_VertBitSet( top, region, &(int){3} );
    num = MR_BitSet_count( MR_VertBitSet_UpcastTo_MR_BitSet( region ) );
    TEST_ASSERT( num == 37 );

    MR_VertBitSet_Destroy( region );
    MR_Mesh_Destroy( mesh );
}
