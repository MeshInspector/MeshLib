#include <MRMesh/MRContoursStitch.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST(MRMesh, cutAlongEdgeLoop)
{
    Mesh mesh = makeCube();
    auto & topology = mesh.topology;
    const auto ueCntA = topology.computeNotLoneUndirectedEdges();

    EdgeLoop c0;
    for ( auto e : leftRing( mesh.topology, 0_f ) )
        c0.push_back( e );
    auto c1 = cutAlongEdgeLoop( mesh.topology, c0 );
    const auto ueCntB = topology.computeNotLoneUndirectedEdges();
    ASSERT_EQ( ueCntB, ueCntA + 3 );

    stitchContours( mesh.topology, c0, c1 );
    const auto ueCntC = topology.computeNotLoneUndirectedEdges();
    ASSERT_EQ( ueCntC, ueCntA );
}

} //namespace MR
