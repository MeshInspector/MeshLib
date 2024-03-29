#include "MRAABBTreePolyline.h"
#include "MRPolyline.h"
#include "MRAABBTreeMaker.h"
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, AABBTreePolyline2 )
{
    Polyline2 polyline;
    polyline.points.vec_ = {
        {0.1f,0.0f},
        {0.1f,1.0f},
        {0.1f,2.0f},
        {0.1f,3.0f},
        {0.1f,4.0f},
        {0.1f,5.0f}
    };

    std::array vs = { 0_v, 1_v, 2_v, 3_v, 4_v, 5_v };
    polyline.topology.makePolyline( vs.data(), vs.size() );

    AABBTreePolyline2 tree( polyline );
    EXPECT_EQ( tree.nodes().size(), getNumNodes( (int)polyline.topology.undirectedEdgeSize() ) );
    Box2f box;
    for ( auto p : polyline.points )
        box.include( p );
    EXPECT_EQ( tree[AABBTreePolyline2::rootNodeId()].box, box );
    EXPECT_TRUE( tree[AABBTreePolyline2::rootNodeId()].l.valid() );
    EXPECT_TRUE( tree[AABBTreePolyline2::rootNodeId()].r.valid() );

    assert( !tree.nodes().empty() );
    auto m = std::move( tree );
    assert( tree.nodes().empty() );
}

} //namespace MR
