#include <MRMesh/MRClosestWeightedPoint.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>

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
    auto distance = [&]( Vector3f loc )
    {
        auto pd = findClosestWeightedMeshPoint( loc, mesh, params );
        assert( !pd.mtp.onEdge( mesh.topology ) );
        return pd.dist;
    };

    {
        params.bidirectionalMode = false;

        params.pointWeight = [&]( VertId ) { return 1; };
        params.maxWeight = 1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ) ), -1 - z, 1e-7f );

        params.pointWeight = [&]( VertId ) { return -1; };
        params.maxWeight = -1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ) ),  1 - z, 1e-7f );
    }

    {
        params.bidirectionalMode = true;

        params.pointWeight = [&]( VertId ) { return 1; };
        params.maxWeight = 1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ) ), -1 + std::abs( z ), 1e-7f );

        params.pointWeight = [&]( VertId ) { return -1; };
        params.maxWeight = -1;
        for ( float z = -2; z <= 2; z += 0.1f )
            EXPECT_NEAR( distance( Vector3f( 0, 0, z ) ),  1 + std::abs( z ), 1e-7f );
    }
}

} //namespace MR
