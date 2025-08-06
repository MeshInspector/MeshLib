#include "TestMacros.h"
#include <MRMeshC/MRBitSet.h>
#include <MRMeshC/MRMesh.h>
#include <MRMeshC/MRMakeSphereMesh.h>
#include <MRMeshC/MRFloatGrid.h>
#include <MRMeshC/MRVDBConversions.h>

MRVdbVolume createVolume(void)
{
    MRSphereParams params;
    params.radius = 1.0f;
    params.numMeshVertices = 3000;

    MRMesh* mesh = mrMakeSphere( &params );

    MRMeshToVolumeSettings settings = mrVdbConversionsMeshToVolumeSettingsNew();
    settings.voxelSize = mrVector3fDiagonal( 0.1f );

    MRVdbVolume volume = mrVdbConversionsMeshToVolume( mesh, &settings, NULL );
    mrMeshFree( mesh );
    return volume;
}

void testVDBConversions( void )
{
    MRVdbVolume volume = createVolume();
    TEST_ASSERT( volume.min > -0.001f && volume.min < 0.001f );
    TEST_ASSERT( volume.max > 2.999f && volume.max < 3.001f );
    TEST_ASSERT( volume.dims.x == 26 );
    TEST_ASSERT( volume.dims.y == 26 );
    TEST_ASSERT( volume.dims.z == 26 );

    MRGridToMeshSettings gridToMeshSettings = mrVdbConversionsGridToMeshSettingsNew();
    gridToMeshSettings.voxelSize = mrVector3fDiagonal( 0.1f );
    gridToMeshSettings.isoValue = 1;

    MRMesh* restored = mrVdbConversionsGridToMesh( volume.data, &gridToMeshSettings, NULL );
    MRBox3f bbox = mrMeshComputeBoundingBox( restored, NULL );
    TEST_ASSERT( bbox.min.x > 0.199f && bbox.min.x < 0.201f );
    TEST_ASSERT( bbox.min.y > 0.199f && bbox.min.y < 0.201f );
    TEST_ASSERT( bbox.min.z > 0.199f && bbox.min.z < 0.201f );
    TEST_ASSERT( bbox.max.x > 2.394f && bbox.max.x < 2.396f );
    TEST_ASSERT( bbox.max.y > 2.394f && bbox.max.y < 2.396f );
    TEST_ASSERT( bbox.max.z > 2.394f && bbox.max.z < 2.396f );

    size_t pointsNum = mrMeshPointsNum( restored );
    TEST_ASSERT( pointsNum == 3748 );

    mrMeshFree( restored );
}

void testUniformResampling( void )
{
    MRVdbVolume volume = createVolume();
    MRFloatGrid* resampledGrid = mrFloatGridResampledUniformly( volume.data, 2.0f, NULL );
    MRVdbVolume resampledVolume = mrVdbConversionsFloatGridToVdbVolume( resampledGrid );
    TEST_ASSERT( resampledVolume.dims.x == 13 );
    TEST_ASSERT( resampledVolume.dims.y == 13 );
    TEST_ASSERT( resampledVolume.dims.z == 13 );
}

void testResampling( void )
{
    MRVdbVolume volume = createVolume();
    MRVector3f voxelScale;
    voxelScale.x = 2.0f;
    voxelScale.y = 1.0f;
    voxelScale.z = 0.5f;

    MRFloatGrid* resampledGrid = mrFloatGridResampled( volume.data, &voxelScale, NULL );
    MRVdbVolume resampledVolume = mrVdbConversionsFloatGridToVdbVolume( resampledGrid );
    TEST_ASSERT( resampledVolume.dims.x == 13 );
    TEST_ASSERT( resampledVolume.dims.y == 27 );
    TEST_ASSERT( resampledVolume.dims.z == 53 );
}

void testCropping( void )
{
    MRVdbVolume volume = createVolume();
    MRBox3i box;
    box.min.x = 2;
    box.min.y = 5;
    box.min.z = 1;
    box.max.x = 18;
    box.max.y = 13;
    box.max.z = 23;

    MRFloatGrid* croppedGrid = mrFloatGridCropped( volume.data, &box, NULL );
    MRVdbVolume croppedVolume = mrVdbConversionsFloatGridToVdbVolume( croppedGrid );
    TEST_ASSERT( croppedVolume.dims.x == 16 );
    TEST_ASSERT( croppedVolume.dims.y == 8 );
    TEST_ASSERT( croppedVolume.dims.z == 22 );
}

void testAccessors( void )
{
    MRVdbVolume volume = createVolume();
    MRVector3i p;
    p.x = 0; p.y = 0; p.z = 0;
    float value = mrFloatGridGetValue( volume.data, &p );
    TEST_ASSERT( value == 3.0f );

    MRVoxelBitSet* region = mrVoxelBitSetNew( volume.dims.x * volume.dims.y * volume.dims.z, false );
    mrBitSetSet( (MRBitSet*)region, 0, true );
    mrFloatGridSetValueForRegion( volume.data, region, 1.0f );
    mrVoxelBitSetFree( region );

    value = mrFloatGridGetValue( volume.data, &p );
    TEST_ASSERT( value == 1.0f );

    mrFloatGridSetValue( volume.data, &p, 2.0f );
    value = mrFloatGridGetValue( volume.data, &p );
    TEST_ASSERT( value == 2.0f );
}
