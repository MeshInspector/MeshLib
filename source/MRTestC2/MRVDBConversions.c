#include "TestMacros.h"
#include <MRCMesh/MRBitSet.h>
#include <MRCMesh/MRMakeSphereMesh.h>
#include <MRCMesh/MRMesh.h>
#include <MRCMesh/MRMeshPart.h>
#include <MRCMesh/MRVector.h>
#include <MRCMisc/expected_MR_Mesh_std_string.h>
#include <MRCMisc/expected_MR_VdbVolume_std_string.h>
#include <MRCMisc/std_shared_ptr_MR_OpenVdbFloatGrid.h>
#include <MRCVoxels/MRFloatGrid.h>
#include <MRCVoxels/MRVDBConversions.h>
#include <MRCVoxels/MRVoxelsVolume.h>

MR_VdbVolume* createVolume(void)
{
    MR_SphereParams* sphereParams = MR_SphereParams_DefaultConstruct();
    MR_SphereParams_Set_radius( sphereParams, 1.0f );
    MR_SphereParams_Set_numMeshVertices( sphereParams, 3000 );

    MR_Mesh* mesh = MR_makeSphere( sphereParams );
    MR_SphereParams_Destroy( sphereParams );

    MR_MeshPart* mp = MR_MeshPart_Construct( mesh, NULL );

    MR_MeshToVolumeParams* mtvParams = MR_MeshToVolumeParams_DefaultConstruct();
    MR_MeshToVolumeParams_Set_voxelSize( mtvParams, MR_Vector3f_diagonal( 0.1f ) );


    MR_expected_MR_VdbVolume_std_string* volumeEx = MR_meshToVolume( mp, mtvParams );

    MR_MeshPart_Destroy( mp );
    MR_MeshToVolumeParams_Destroy( mtvParams );
    MR_Mesh_Destroy( mesh );

    MR_VdbVolume *volume = MR_expected_MR_VdbVolume_std_string_GetMutableValue( volumeEx );
    TEST_ASSERT( volume )
    return volume;
}

void testVDBConversions( void )
{
    MR_VdbVolume* volume = createVolume();

    const MR_Box1f* box = MR_VdbVolume_UpcastTo_MR_Box1f( volume );
    TEST_ASSERT( box->min > -0.001f && box->min < 0.001f );
    TEST_ASSERT( box->max > 2.999f && box->max < 3.001f );
    const MR_VoxelsVolume_MR_FloatGrid* baseVolume = MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume );
    const MR_Vector3i* dims = MR_VoxelsVolume_MR_FloatGrid_Get_dims( baseVolume );
    TEST_ASSERT( dims->x == 26 );
    TEST_ASSERT( dims->y == 26 );
    TEST_ASSERT( dims->z == 26 );

    MR_GridToMeshSettings* gridToMeshSettings = MR_GridToMeshSettings_DefaultConstruct();
    MR_GridToMeshSettings_Set_voxelSize( gridToMeshSettings, MR_Vector3f_diagonal( 0.1f ) );
    MR_GridToMeshSettings_Set_isoValue( gridToMeshSettings, 1 );

    MR_expected_MR_Mesh_std_string* restoredEx = MR_gridToMesh_const_MR_FloatGrid_ref( MR_VoxelsVolume_MR_FloatGrid_Get_data( baseVolume ), gridToMeshSettings );
    const MR_Mesh* restored = MR_expected_MR_Mesh_std_string_GetValue( restoredEx );
    MR_Box3f bbox = MR_Mesh_computeBoundingBox_1( restored, NULL );
    TEST_ASSERT( bbox.min.x > 0.199f && bbox.min.x < 0.201f );
    TEST_ASSERT( bbox.min.y > 0.199f && bbox.min.y < 0.201f );
    TEST_ASSERT( bbox.min.z > 0.199f && bbox.min.z < 0.201f );
    TEST_ASSERT( bbox.max.x > 2.394f && bbox.max.x < 2.396f );
    TEST_ASSERT( bbox.max.y > 2.394f && bbox.max.y < 2.396f );
    TEST_ASSERT( bbox.max.z > 2.394f && bbox.max.z < 2.396f );

    size_t pointsNum = MR_VertCoords_size( MR_Mesh_Get_points( restored ) );
    TEST_ASSERT( pointsNum == 3748 );

    MR_expected_MR_Mesh_std_string_Destroy( restoredEx );

    MR_VdbVolume_Destroy( volume );
}

void testUniformResampling( void )
{
    MR_VdbVolume* volume = createVolume();
    MR_FloatGrid* resampledGrid = MR_resampled_float( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), 2.0f, MR_PassBy_DefaultArgument, NULL );
    MR_VdbVolume* resampledVolume = MR_floatGridToVdbVolume( MR_PassBy_Move, resampledGrid );
    MR_FloatGrid_Destroy( resampledGrid );

    const MR_VoxelsVolume_MR_FloatGrid* baseResampledVolume = MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid( resampledVolume );
    const MR_Vector3i* dims = MR_VoxelsVolume_MR_FloatGrid_Get_dims( baseResampledVolume );

    TEST_ASSERT( dims->x == 13 );
    TEST_ASSERT( dims->y == 13 );
    TEST_ASSERT( dims->z == 13 );

    MR_VdbVolume_Destroy( resampledVolume );
    MR_VdbVolume_Destroy( volume );
}

void testResampling( void )
{
    MR_VdbVolume* volume = createVolume();
    MR_Vector3f voxelScale;
    voxelScale.x = 2.0f;
    voxelScale.y = 1.0f;
    voxelScale.z = 0.5f;

    MR_FloatGrid* resampledGrid = MR_resampled_MR_Vector3f( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &voxelScale, MR_PassBy_DefaultArgument, NULL );
    MR_VdbVolume* resampledVolume = MR_floatGridToVdbVolume( MR_PassBy_Move, resampledGrid );
    MR_FloatGrid_Destroy( resampledGrid );

    const MR_VoxelsVolume_MR_FloatGrid* baseResampledVolume = MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid( resampledVolume );
    const MR_Vector3i* dims = MR_VoxelsVolume_MR_FloatGrid_Get_dims( baseResampledVolume );

    TEST_ASSERT( dims->x == 13 );
    TEST_ASSERT( dims->y == 27 );
    TEST_ASSERT( dims->z == 53 );

    MR_VdbVolume_Destroy( resampledVolume );
    MR_VdbVolume_Destroy( volume );
}

void testCropping( void )
{
    MR_VdbVolume* volume = createVolume();
    MR_Box3i box;
    box.min.x = 2;
    box.min.y = 5;
    box.min.z = 1;
    box.max.x = 18;
    box.max.y = 13;
    box.max.z = 23;

    MR_FloatGrid* croppedGrid = MR_cropped( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &box, MR_PassBy_DefaultArgument, NULL );

    MR_VdbVolume* croppedVolume = MR_floatGridToVdbVolume( MR_PassBy_Move, croppedGrid );
    MR_FloatGrid_Destroy( croppedGrid );

    const MR_VoxelsVolume_MR_FloatGrid* baseCroppedVolume = MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid( croppedVolume );
    const MR_Vector3i* dims = MR_VoxelsVolume_MR_FloatGrid_Get_dims( baseCroppedVolume );

    TEST_ASSERT( dims->x == 16 );
    TEST_ASSERT( dims->y == 8 );
    TEST_ASSERT( dims->z == 22 );

    MR_VdbVolume_Destroy( croppedVolume );
    MR_VdbVolume_Destroy( volume );
}

void testAccessors( void )
{
    MR_VdbVolume* volume = createVolume();
    MR_Vector3i p;
    p.x = 0; p.y = 0; p.z = 0;
    float value = MR_getValue( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &p );
    TEST_ASSERT( value == 3.0f );

    const MR_VoxelsVolume_MR_FloatGrid* baseVolume = MR_VdbVolume_UpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume );
    const MR_Vector3i* dims = MR_VoxelsVolume_MR_FloatGrid_Get_dims( baseVolume );

    MR_VoxelBitSet* region = MR_VoxelBitSet_DefaultConstruct();
    MR_BitSet_resize( MR_VoxelBitSet_MutableUpcastTo_MR_BitSet( region ), dims->x * dims->y * dims->z, NULL );

    MR_BitSet_set_2( MR_VoxelBitSet_MutableUpcastTo_MR_BitSet( region ), 0, true );

    MR_setValue_MR_VoxelBitSet( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), region, 1.0f );
    MR_VoxelBitSet_Destroy( region );

    value = MR_getValue( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &p );
    TEST_ASSERT( value == 1.0f );

    MR_setValue_MR_Vector3i( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &p, 2.0f );
    value = MR_getValue( MR_VoxelsVolume_MR_FloatGrid_GetMutable_data( MR_VdbVolume_MutableUpcastTo_MR_VoxelsVolume_MR_FloatGrid( volume ) ), &p );
    TEST_ASSERT( value == 2.0f );

    MR_VdbVolume_Destroy( volume );
}
