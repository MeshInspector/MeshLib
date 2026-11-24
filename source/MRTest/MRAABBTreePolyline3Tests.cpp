#include <MRMesh/MRAABBTreePolyline.h>
#include <MRMesh/MRPolyline.h>
#include <MRMesh/MRAABBTreeMaker.h>
#include <MRMesh/MRBox.h>
#include <MRMesh/MRVector3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, AABBTreePolyline3 )
{
    Polyline3 polyline;
    polyline.points = {
        {0.0f,0.1f,0.0f},
        {0.0f,0.1f,1.0f},
        {0.0f,0.1f,2.0f},
        {0.0f,0.1f,3.0f},
        {0.0f,0.1f,4.0f},
        {0.0f,0.1f,5.0f}
    };

    std::array vs = { 0_v, 1_v, 2_v, 3_v, 4_v, 5_v };
    polyline.topology.makePolyline( vs.data(), vs.size() );

    AABBTreePolyline3 tree( polyline );
    EXPECT_EQ( tree.nodes().size(), getNumNodes( (int)polyline.topology.undirectedEdgeSize() ) );
    Box3f box;
    for ( auto p : polyline.points )
        box.include( p );
    EXPECT_EQ( tree[tree.rootNodeId()].box, box );
    EXPECT_EQ( tree.getBoundingBox(), box );
    EXPECT_EQ( tree.numLeaves(), polyline.topology.undirectedEdgeSize() );

    assert( !tree.empty() );
    auto m = std::move( tree );
    assert( tree.empty() );
}

} //namespace MR
