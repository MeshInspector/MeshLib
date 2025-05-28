#include <MRMesh/MRClosestWeightedPoint.h>
#include <MRMesh/MRPointCloud.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRVoxels/MRWeightedPointsShell.h>
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

        params.maxBidirDist = 0.5f;
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

        params.maxBidirDist = 1.5f;
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
        return pd.dist();
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

TEST( MRMesh, weightedMeshShell )
{
    auto cube = makeCube();
    auto rot = Matrix3f::rotation( { 1.f, 1.f, 1.f }, PI_F / 6.f );
    for ( auto& pt : cube.points )
        pt = rot * pt;

    auto offCube = WeightedShell::meshShell( cube, WeightedShell::ParametersRegions{ { 0.08f, 0.04f, 3.f }, {}, 0.f, false } );
    ASSERT_TRUE( offCube );
    EXPECT_EQ( MeshComponents::getNumComponents( *offCube ), 1 );
    EXPECT_EQ( offCube->topology.findNumHoles(), 0 );
}

static void testClosestWeightedMeshPointContinuity( bool bidir )
{
    auto cube = makeCube();
    DistanceFromWeightedPointsComputeParams params;
    auto distance = [&]( Vector3f loc )
    {
        auto pd = findClosestWeightedMeshPoint( loc, cube, params );
        return pd.dist();
    };

    params.pointWeight = [] ( VertId v ) { return (int)v * 0.1f; };
    params.maxWeight = 0.7f;
    params.bidirectionalMode = bidir;

    for ( auto v : cube.topology.getValidVerts() )
        EXPECT_NEAR( distance( cube.points[v] ), -params.pointWeight( v ), 1e-6f );

    constexpr float step = 0.05f;
    constexpr float gradStep = 0.001f;
    constexpr float rangeMin = -0.6f;
    constexpr float rangeMax =  0.6f;
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

TEST( MRMesh, findClosestWeightedMeshPointContinuity )
{
    testClosestWeightedMeshPointContinuity( false );
    testClosestWeightedMeshPointContinuity( true );
}

TEST( MRMesh, WeightedClosed )
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v }
    };
    VertCoords vs{
        {-1, -1, 0 },  //0_v
        {-1,  1, 0 },  //1_v
        { 1,  0, 0 },  //2_v
        {-1,  1, 3 },  //0_v
        {-1, -1, 3 },  //1_v
        { 1,  0, 3 }   //2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( vs ), t );

    VertScalars weights {
        2, 2, 2, 0, 0, 0    // TODO: fix method for 4,4,4,0,0,0
    };

    DistanceFromWeightedPointsComputeParams params;
    params.pointWeight = [&]( VertId v ) { return weights[v]; };
    params.maxWeight = 2;
    params.bidirectionalMode = false;

    auto smartDistance = [&]( Vector3f loc )
    {
        auto pd = findClosestWeightedMeshPoint( loc, mesh, params );
        assert( !pd.mtp.onEdge( mesh.topology ) );
        return pd.dist();
    };

//    std::ofstream f( "/tmp/test.csv" );
//    f << "z,d\n";
//    for ( float z = -1; z <= 4; z += 0.1f )
//        f << z << ',' << smartDistance( Vector3f( 0, 0, z ) ) << '\n';
//    f.close();

    float maxDiff = 0.f;
    for ( float z = -1; z <= 4; z += 0.1f )
        maxDiff = std::max( maxDiff, std::abs( smartDistance( Vector3f{ 0.f, 0.f, z - 0.5f } ) - smartDistance( Vector3f{ 0.f, 0.f, z + 0.5f } ) ) );
    ASSERT_FLOAT_EQ( maxDiff, 1.f );
}

TEST( MRMesh, findClosestWeightedMeshPointSharpAngle )
{
    Triangulation t{
        { 0_v, 2_v, 1_v },
        { 0_v, 1_v, 3_v }
    };
    VertCoords vs{
        { 1, 0, 0 },        //0_v
        { 0, 0, 0 },        //1_v
        { 0.5, 0.5, 1 },    //2_v
        { -0.5, -0.5, 1 }   //3_v
    };
    auto mesh = Mesh::fromTriangles( std::move( vs ), t );

    DistanceFromWeightedPointsComputeParams params;
    VertScalars weights{ 0.2f, 0.4f, 0.1f, 0.3f };
    params.pointWeight = [&]( VertId v ) { return weights[v]; };
    params.maxWeight = 0;
    params.bidirectionalMode = false;

    auto smartDistance = [&]( Vector3f loc )
    {
        auto pd = findClosestWeightedMeshPoint( loc, mesh, params );
        return pd.dist();
    };

//    std::ofstream f( "/tmp/test.csv" );
//    f << "z,d\n";
//    for ( float z = -0.5; z <= 0.5; z += 0.05f )
//        f << z << ',' << smartDistance( Vector3f( 0.1, 0.1, z ) ) << '\n';
//    f.close();

    float maxDiff = 0.f;
    for ( float z = -0.5f; z <= 0.5f; z += 0.05f )
        maxDiff = std::max( maxDiff, std::abs( smartDistance( Vector3f{ 0.1f, 0.1f, z - 0.025f } ) - smartDistance( Vector3f{ 0.1f, 0.1f, z + 0.025f } ) ) );
    ASSERT_LE( maxDiff, 0.05f );
}

} //namespace MR
