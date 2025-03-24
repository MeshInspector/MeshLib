#include <MRMesh/MRPointCloudVariadicOffset.h>
#include <MRMesh/MRPointCloud.h>
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
        VariadicOffsetParams params{ .maxWeightGrad = 0 };
        auto ws = []( VertId ) { return 0.f; };
        auto res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), ws, params );
        ASSERT_TRUE( res.valid() );
        ASSERT_EQ( res.vId,  0_v );
        ASSERT_EQ( res.dist, 1 );

        params.maxDistance = 0.5f;
        res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), ws, params );
        ASSERT_FALSE( res.valid() );
    }

    {
        VariadicOffsetParams params{ .maxWeightGrad = 0.5f };
        auto ws = []( VertId v ) { return v == 0_v ? -1.5f : 0.f; };
        auto res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), ws, params );
        ASSERT_TRUE( res.valid() );
        ASSERT_EQ( res.vId,  1_v );
        ASSERT_EQ( res.dist, 2 );

        params.maxDistance = 1.5f;
        res = findClosestWeightedPoint( Vector3f( 1, 0, 0 ), pc.getAABBTree(), ws, params );
        ASSERT_FALSE( res.valid() );
    }
}

} //namespace MR
