#ifndef MESHLIB_NO_VOXELS

#include "MRVoxels/MRBoolean.h"
#include "MRVoxels/MRMeshToDistanceVolume.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRVolumeIndexer.h"
#include "MRMesh/MRMeshDistance.h"
#include "MRMesh/MRTorus.h"
#include "MRMesh/MRMesh.h"
#include <gtest/gtest.h>
#include "MRPch/MRSpdlog.h"
#include "MRPch/MRTBB.h"
#include <openvdb/version.h>

namespace MR
{

// below test crashed on Fedora 40 and 41 because of incompatibility between VDB and TBB
TEST( MRMesh, VersionVDB )
{
#if defined TBB_VERSION_PATCH
    spdlog::info( "TBB version: {}.{}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR, TBB_VERSION_PATCH );
#else
    spdlog::info( "TBB version: {}.{}", TBB_VERSION_MAJOR, TBB_VERSION_MINOR );
#endif
    spdlog::info( "OpenVDB version: {}", OPENVDB_LIBRARY_VERSION_STRING );
}

TEST( MRMesh, MeshVoxelsConverterSelfIntersections )
{
    auto torus = makeTorusWithSelfIntersections( 2.f, 1.f, 10, 10 );
    MeshVoxelsConverter converter;
    converter.voxelSize = 0.1f;
    auto grid = converter( torus );
    torus = converter( grid );
    ASSERT_GT( torus.volume(), 0.f );
}

TEST( MRMesh, MakeInsideMeshVolume )
{
    const auto sphere = makeSphere( { .radius = 1.0f, .numMeshVertices = 10000 } );
    DistanceVolumeParams params;
    params.origin = Vector3f::diagonal( -1.5f );
    params.voxelSize = Vector3f::diagonal( 0.1f );
    params.dimensions = Vector3i::diagonal( 30 );
    const auto vol = makeInsideMeshVolume( sphere, params );
    ASSERT_TRUE( vol.has_value() );
    EXPECT_EQ( vol->dims, params.dimensions );

    const VolumeIndexer indexer( vol->dims );
    int numChecked = 0;
    for ( size_t n = 0; n < indexer.size(); ++n )
    {
        const VoxelId i( n );
        const auto center = params.origin + mult( params.voxelSize, Vector3f( indexer.toPos( i ) ) + Vector3f::diagonal( 0.5f ) );
        const auto r = center.length();
        if ( std::abs( r - 1.0f ) < 0.02f )
            continue; // too close to the surface approximated by triangles
        EXPECT_EQ( vol->data.test( i ), r < 1.0f );
        ++numChecked;
    }
    EXPECT_GT( numChecked, 25000 );
}

TEST( MRMesh, MeshToDistanceVolumeWindingRule )
{
    const auto torus = makeTorus( 1.0f, 0.4f, 64, 32 );
    MeshToDistanceVolumeParams params;
    params.vol.origin = Vector3f( -1.5f, -1.5f, -0.5f );
    params.vol.voxelSize = Vector3f::diagonal( 0.05f );
    params.vol.dimensions = Vector3i( 60, 60, 20 );
    params.dist.signMode = SignDetectionMode::WindingRule;
    const auto vol = meshToDistanceVolume( torus, params );
    ASSERT_TRUE( vol.has_value() );

    // the signs are found by one ray per row of voxels, and must be the same as from a ray per voxel
    const VolumeIndexer indexer( vol->dims );
    int numNegative = 0;
    for ( size_t n = 0; n < indexer.size(); ++n )
    {
        const VoxelId i( n );
        const auto center = params.vol.origin + mult( params.vol.voxelSize, Vector3f( indexer.toPos( i ) ) + Vector3f::diagonal( 0.5f ) );
        const auto dist = signedDistanceToMesh( torus, center, params.dist );
        ASSERT_TRUE( dist.has_value() );
        if ( std::abs( *dist ) < 1e-4f )
            continue; // the sign of a point on the surface is uncertain
        EXPECT_EQ( vol->data[i], *dist );
        if ( *dist < 0 )
            ++numNegative;
    }
    EXPECT_GT( numNegative, 5000 );
}

} //namespace MR

#endif //!MESHLIB_NO_VOXELS
