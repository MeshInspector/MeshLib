#include <MRMesh/MRClosestWeightedPoint.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRVoxels/MRWeightedPointsShell.h>
#include <MRMesh/MRMeshComponents.h>

namespace MR
{

TEST( MRMesh, findClosestWeightedPoint )
{
    PointCloud pc;
    pc.points.push_back( Vector3f( 0, 0, 0 ) );
    pc.points.push_back( Vector3f( 3, 0, 0 ) );
    pc.validPoints.resize( 2, true );

    {
        DistanceFromWeightedPointsComputeParams params
        { {
            .pointWeight = []( VertId ) { return 0.f; },
            .maxWeightGrad = 0
        } };
        auto res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), params );
        ASSERT_TRUE( res.valid() );
        ASSERT_EQ( res.vId,  0_v );
        ASSERT_EQ( res.dist, 1 );

        params.maxDistance = 0.5f;
        res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), params );
        ASSERT_FALSE( res.valid() );
    }

    {
        DistanceFromWeightedPointsComputeParams params
        { {
            .pointWeight = []( VertId v ) { return v == 0_v ? -1.5f : 0.f; },
            .maxWeightGrad = 0.5f
        } };
        auto res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), params );
        ASSERT_TRUE( res.valid() );
        ASSERT_EQ( res.vId,  1_v );
        ASSERT_EQ( res.dist, 2 );

        params.maxDistance = 1.5f;
        res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), params );
        ASSERT_FALSE( res.valid() );
    }
}

TEST( MRMesh, findClosestWeightedMeshPoint )
{
    Triangulation t{
        { 0_v, 1_v, 2_v }
    };
    VertCoords vs{
        {-1, -1, 0 },  //0_v
        {-1,  1, 0 },  //1_v
        { 1,  0, 0 },  //2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( vs ), t );

    DistanceFromWeightedPointsComputeParams params;
    auto distance = [&]( Vector3f loc, bool bidirectional )
    {
        auto pd = findClosestWeightedMeshPoint( loc, mesh, params );
        assert( !pd.mtp.onEdge( mesh.topology ) );
        return pd.weightedDist( bidirectional );
    };

    {
        params.bidirectionalMode = false;

        params.pointWeight = [&]( VertId ) { return 1; };
        params.maxWeight = 1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ), params.bidirectionalMode ), -1 - z, 1e-7f );

        params.pointWeight = [&]( VertId ) { return -1; };
        params.maxWeight = -1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ), params.bidirectionalMode ),  1 - z, 1e-7f );
    }

    {
        params.bidirectionalMode = true;

        params.pointWeight = [&]( VertId ) { return 1; };
        params.maxWeight = 1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ), params.bidirectionalMode ), -1 + std::abs( z ), 1e-7f );

        params.pointWeight = [&]( VertId ) { return -1; };
        params.maxWeight = -1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ), params.bidirectionalMode ),  1 + std::abs( z ), 1e-7f );
    }
}

TEST( MRMesh, weightedMeshShell )
{
    auto cube = makeCube();
    auto rot = Matrix3f::rotation( { 1.f, 1.f, 1.f }, PI_F / 6.f );
    for ( auto& pt : cube.points )
        pt = rot * pt;

    ASSERT_TRUE( remesh( cube, { .maxEdgeSplits = 10000 } ) );

    auto offCube = weightedMeshShell( cube, WeightedPointsShellParametersRegions{ { 0.02, 0.01, 10.f }, {}, 0.f, false } );
    ASSERT_TRUE( offCube );
    auto components = MeshComponents::getAllComponents( MeshPart{ *offCube } );
    EXPECT_EQ( components.size(), 1 );
    EXPECT_EQ( offCube->topology.findNumHoles(), 0 );
}

} //namespace MR
