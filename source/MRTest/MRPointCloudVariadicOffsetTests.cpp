#include <MRMesh/MRClosestWeightedPoint.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRParallelFor.h>

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

TEST( MRMesh, findClosestWeightedMeshPointContinuity )
{
    auto cube = makeCube();
    DistanceFromWeightedPointsComputeParams params;
    auto distance = [&]( Vector3f loc )
    {
        auto pd = findClosestWeightedMeshPoint( loc, cube, params );
        return pd.dist;
    };

    params.maxWeight = 0.1f;
    params.bidirectionalMode = false;
    //params.pointWeight = [v0 = cube.topology.getValidVerts().find_first()] ( VertId v ) { return ( v == v0 ) ? 0.1f : 0.f; };
    params.pointWeight = [] ( VertId ) { return 0.1f; };

    constexpr float step = 0.05f;
    constexpr float gradStep = 0.001f;
    constexpr float rangeMin = -0.8f;
    constexpr float rangeMax = 0.8f;
    std::vector<float> zVals;
    for ( float z = rangeMin; z < rangeMax; z += step )
        zVals.push_back( z );

    // check that distance is continuous
    tbb::enumerable_thread_specific<float> threadMaxGrad;
    ParallelFor( zVals, [&] ( size_t iz ) {
        float z = zVals[iz];
        float localMaxGrad = 0.f;
        for ( float y = rangeMin; y < rangeMax; y += step )
        {
            for ( float x = rangeMin; x < rangeMax; x += step )
            {
                Vector3f pt{ x, y, z };
                Vector3f grad{
                        ( distance( pt + Vector3f::plusX() * gradStep ) - distance( pt - Vector3f::plusX() * gradStep ) ) / ( 2.f * gradStep ),
                        ( distance( pt + Vector3f::plusY() * gradStep ) - distance( pt - Vector3f::plusY() * gradStep ) ) / ( 2.f * gradStep ),
                        ( distance( pt + Vector3f::plusZ() * gradStep ) - distance( pt - Vector3f::plusZ() * gradStep ) ) / ( 2.f * gradStep )
                };
                localMaxGrad = std::max( localMaxGrad, grad.length() );
            }
        }
        threadMaxGrad.local() = localMaxGrad;
    } );

    float maxGrad = 0.f;
    for ( float val : threadMaxGrad )
        maxGrad = std::max( maxGrad, val );

    EXPECT_NEAR( maxGrad, 1.f, 0.1f ); // gradient should be 1 as distance should change linearly
}

} //namespace MR
