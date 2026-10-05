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
    using MakeInside = Expected<SimpleBinaryVolume>( * )( const MeshPart&, const DistanceVolumeParams& );
    for ( MakeInside makeInside : {
        MakeInside( [] ( const MeshPart& mp, const DistanceVolumeParams& p ) { return makeInsideMeshVolume( mp, p, InsideMeshRule::OddCrossings ); } ),
        MakeInside( [] ( const MeshPart& mp, const DistanceVolumeParams& p ) { return makeInsideMeshVolume( mp, p, InsideMeshRule::PositiveWinding ); } ),
        MakeInside( [] ( const MeshPart& mp, const DistanceVolumeParams& p ) { return makeInsideMeshVolumeVdb( mp, p ); } ) } )
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

TEST( MRMesh, MakeInsideMeshVolumeRules )
{
    auto makeSphereAt = [] ( float radius, const Vector3f& center, bool flip )
    {
        auto res = makeSphere( { .radius = radius, .numMeshVertices = 3000 } );
        res.transform( AffineXf3f::translation( center ) );
        if ( flip )
            res.topology.flipOrientation();
        return res;
    };
    DistanceVolumeParams params;
    params.origin = Vector3f::diagonal( -2.5f );
    params.voxelSize = Vector3f::diagonal( 0.1f );
    params.dimensions = Vector3i::diagonal( 50 );
    const VolumeIndexer indexer( params.dimensions );

    // returns { odd crossings, positive winding, OpenVDB } inside-values of the voxel containing point p
    auto probe = [&] ( const Mesh& mesh, const Vector3f& p )
    {
        const auto vox = indexer.toVoxelId( Vector3i( div( p - params.origin, params.voxelSize ) ) );
        return std::array<bool, 3>{
            makeInsideMeshVolume( mesh, params, InsideMeshRule::OddCrossings )->data.test( vox ),
            makeInsideMeshVolume( mesh, params, InsideMeshRule::PositiveWinding )->data.test( vox ),
            makeInsideMeshVolumeVdb( mesh, params )->data.test( vox ) };
    };
    using B = std::array<bool, 3>;

    // two overlapping spheres, not united
    auto twoSpheres = makeSphereAt( 1, { -0.5f, 0, 0 }, false );
    twoSpheres.addMesh( makeSphereAt( 1, { 0.5f, 0, 0 }, false ) );
    EXPECT_EQ( probe( twoSpheres, { 0.05f, 0.05f, 0.05f } ), B( { false, true, true } ) ); // in both spheres
    EXPECT_EQ( probe( twoSpheres, { 1.15f, 0.05f, 0.05f } ), B( { true, true, true } ) ); // in one sphere
    EXPECT_EQ( probe( twoSpheres, { 2.05f, 0.05f, 0.05f } ), B( { false, false, false } ) ); // outside

    // hollow shell: the inner sphere looks inside the cavity, or wrongly outside
    for ( bool flipInner : { true, false } )
    {
        auto shell = makeSphereAt( 1.8f, {}, false );
        shell.addMesh( makeSphereAt( 1, {}, flipInner ) );
        EXPECT_EQ( probe( shell, { 0.05f, 0.05f, 0.05f } ), B( { false, !flipInner, true } ) ); // in the cavity
        EXPECT_EQ( probe( shell, { 1.45f, 0.05f, 0.05f } ), B( { true, true, true } ) ); // in the wall
    }
}

} //namespace MR

#endif //!MESHLIB_NO_VOXELS
