#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, splitEdge )
{
    MeshTopology topology;
    EdgeId a = topology.makeEdge();
    EdgeId b = topology.makeEdge();
    EdgeId c = topology.makeEdge();
    EdgeId d = topology.makeEdge();
    topology.splice( b, a.sym() );
    topology.splice( c, b.sym() );
    topology.splice( d, c.sym() );
    topology.splice( a, d.sym() );
    EXPECT_TRUE( topology.isLeftQuad( a ) );
    EXPECT_TRUE( topology.isLeftQuad( a.sym() ) );
    topology.setLeft( a, topology.addFaceId() );
    EdgeId a0 = topology.splitEdge( a );
    EXPECT_TRUE( topology.isLeftTri( a0 ) );
    EXPECT_TRUE( topology.isLeftQuad( a ) );
    EXPECT_EQ( topology.getLeftDegree( a.sym() ), 5 );
    EdgeId a1 = topology.splitEdge( a.sym() );
    EXPECT_TRUE( topology.isLeftTri( a1.sym() ) );
    EXPECT_TRUE( topology.isLeftQuad( a ) );
    EXPECT_EQ( topology.getLeftDegree( a.sym() ), 6 );
}

} //namespace MR
