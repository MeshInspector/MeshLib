#include <MRMesh/MREdgeLengthMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST(MRMesh, EdgeLengthMesh)
{
    auto mesh = makeCube();
    auto elMesh = EdgeLengthMesh::fromMesh( mesh );

    const auto & meshTopology = mesh.topology;
    for ( auto ue = 0_ue; ue < meshTopology.undirectedEdgeSize(); ++ue )
    {
        if ( meshTopology.isLoneEdge( ue ) )
            continue;
        auto u = mesh.cotan( ue );
        auto v = elMesh.cotan( ue );
        EXPECT_NEAR( u, v, 1e-6f );
    }
}

} //namespace MR
