#ifndef MESHLIB_NO_VOXELS

#include "MRVoxels/MRBoolean.h"
#include "MRVoxels/MRMeshToDistanceVolume.h"
#include "MRMesh/MRMakeSphereMesh.h"
#include "MRMesh/MRVolumeIndexer.h"
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
    for ( auto makeInside : { &makeInsideMeshVolume, &makeInsideMeshVolumeVdb } )
    {
        const auto vol = makeInside( sphere, params );
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
}

} //namespace MR

#endif //!MESHLIB_NO_VOXELS
