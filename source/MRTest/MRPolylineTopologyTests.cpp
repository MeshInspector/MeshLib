#include <MRMesh/MRPolylineTopology.h>
#include <MRMesh/MRId.h>
#include "MRGTest.h"

namespace MR
{

TEST( MRMesh, PolylineTopology )
{
    PolylineTopology t;
    VertId vs[4] = { 0_v, 1_v, 2_v, 0_v };
    t.makePolyline( vs, 4 );
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 0_v );
    EXPECT_EQ( t.dest( 0_e ), 1_v );

    t.flip();
    EXPECT_TRUE( t.checkValidity() );
    EXPECT_TRUE( t.isConsistentlyOriented() );
    EXPECT_EQ( t.org( 0_e ), 1_v );
    EXPECT_EQ( t.dest( 0_e ), 0_v );

    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 3 );

    t.deleteEdge( 0_ue );
    EXPECT_EQ( t.numValidVerts(), 3 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 2 );

    t.deleteEdge( 1_ue );
    EXPECT_EQ( t.numValidVerts(), 2 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 1 );

    t.deleteEdge( 2_ue );
    EXPECT_EQ( t.numValidVerts(), 0 );
    EXPECT_EQ( t.computeNotLoneUndirectedEdges(), 0 );
}

} //namespace MR
