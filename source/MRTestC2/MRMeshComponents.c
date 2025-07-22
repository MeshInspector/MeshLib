#include "TestMacros.h"

#include "MRMeshComponents.h"

#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRCube.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshBoolean.h>
#include <MRCMesh/MRMeshComponents.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRVector.h>
#include <MRCMesh/MRVector3.h>
#include <MRCMisc/std_pair_MR_Face2RegionMap_int32_t.h>
#include <MRCMisc/std_pair_MR_FaceBitSet_int32_t.h>

typedef struct CreatedMesh
{
    MR_Mesh* mesh;
    MR_MeshPart* part;
} CreatedMesh;

CreatedMesh createMesh( void )
{
    const MR_Vector3f meshASize = MR_Vector3f_diagonal( 1.0f );
    const MR_Vector3f meshABase = MR_Vector3f_diagonal( -0.5f );

    const MR_Vector3f meshBSize = MR_Vector3f_diagonal( 0.1f );
    const MR_Vector3f meshBBase = MR_Vector3f_diagonal( 1.0f );

    MR_Mesh* meshA = MR_makeCube( &meshASize, &meshABase );
    MR_Mesh* meshB = MR_makeCube( &meshBSize, &meshBBase );

    MR_BooleanResult* resultAB = MR_boolean_4_const_MR_Mesh_ref( meshA, meshB, MR_BooleanOperation_Union, NULL );
    MR_Mesh_Destroy( meshA );
    MR_Mesh_Destroy( meshB );

    CreatedMesh ret;
    ret.mesh = MR_Mesh_ConstructFromAnother( MR_PassBy_Copy, MR_BooleanResult_GetMutable_mesh( resultAB ) );
    MR_BooleanResult_Destroy( resultAB );

    ret.part = MR_MeshPart_Construct( ret.mesh, NULL );

    return ret;
}

void destroyMesh( CreatedMesh target )
{
    MR_MeshPart_Destroy( target.part );
    MR_Mesh_Destroy( target.mesh );
}

void testComponentsMap( void )
{
    CreatedMesh m = createMesh();

    MR_std_pair_MR_Face2RegionMap_int32_t* map = MR_MeshComponents_getAllComponentsMap( m.part, NULL, NULL );
    TEST_ASSERT( *MR_std_pair_MR_Face2RegionMap_int32_t_Second( map ) == 2 );
    TEST_ASSERT( MR_Face2RegionMap_size( MR_std_pair_MR_Face2RegionMap_int32_t_First( map ) ) == 24 );
    TEST_ASSERT( MR_Face2RegionMap_index_const( MR_std_pair_MR_Face2RegionMap_int32_t_First( map ), (MR_FaceId){0} )->id_ == 0 );
    TEST_ASSERT( MR_Face2RegionMap_index_const( MR_std_pair_MR_Face2RegionMap_int32_t_First( map ), (MR_FaceId){12} )->id_ == 1 );

    MR_std_pair_MR_Face2RegionMap_int32_t_Destroy( map );

    destroyMesh( m );
}

void testLargeRegions( void )
{
    CreatedMesh m = createMesh();

    MR_std_pair_MR_Face2RegionMap_int32_t* map = MR_MeshComponents_getAllComponentsMap( m.part, NULL, NULL );
    MR_std_pair_MR_FaceBitSet_int32_t* regions = MR_MeshComponents_getLargeByAreaRegions( m.part, MR_std_pair_MR_Face2RegionMap_int32_t_First( map ), *MR_std_pair_MR_Face2RegionMap_int32_t_Second( map ), 0.1f );

    TEST_ASSERT( *MR_std_pair_MR_FaceBitSet_int32_t_Second( regions ) == 1 );
    TEST_ASSERT( MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( MR_std_pair_MR_FaceBitSet_int32_t_First( regions ) ), 0 ) );
    TEST_ASSERT( !MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( MR_std_pair_MR_FaceBitSet_int32_t_First( regions ) ), 12 ) );

    MR_std_pair_MR_Face2RegionMap_int32_t_Destroy( map );

    destroyMesh( m );
}

void testLargeComponents( void )
{
    CreatedMesh m = createMesh();

    MR_FaceBitSet* components = MR_MeshComponents_getLargeByAreaComponents_3( m.part, 0.1f, NULL );
    TEST_ASSERT( MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( components ), 0 ) );
    TEST_ASSERT( !MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( components ), 12 ) );

    MR_FaceBitSet_Destroy( components );

    destroyMesh( m );
}

void testLargestComponent( void )
{
    CreatedMesh m = createMesh();

    int smallerComponents = 0;
    MR_FaceBitSet* largestComponent = MR_MeshComponents_getLargestComponent( m.part, NULL, NULL, &(float){0.1f}, &smallerComponents );
    TEST_ASSERT( MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( largestComponent ), 0 ) );
    TEST_ASSERT( !MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( largestComponent ), 12 ) );
    TEST_ASSERT( smallerComponents == 1 );

    MR_FaceBitSet_Destroy( largestComponent );

    destroyMesh( m );
}

void testGetComponent( void )
{
    CreatedMesh m = createMesh();
    MR_FaceId face;
    face.id_ = 12;

    MR_FaceBitSet* component = MR_MeshComponents_getComponent( m.part, face, NULL, NULL );

    TEST_ASSERT( !MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( component ), 0 ) );
    TEST_ASSERT( MR_BitSet_test( MR_FaceBitSet_UpcastTo_MR_BitSet( component ), 12 ) );
    MR_FaceBitSet_Destroy( component );

    destroyMesh( m );
}
