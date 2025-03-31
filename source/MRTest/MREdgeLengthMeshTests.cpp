#include <MRMesh/MREdgeLengthMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, EdgeLengthMesh )
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

    {
        const EdgeId e = 22_e;
        EXPECT_NEAR( elMesh.edgeLengths[e], 1.f, 1e-6f );
        EXPECT_EQ( elMesh.topology.org( e ), 7_v );
        EXPECT_EQ( elMesh.topology.dest( e ), 4_v );

        EXPECT_TRUE( elMesh.flipEdge( e ) );
        EXPECT_NEAR( elMesh.edgeLengths[e], std::sqrt( 5.f ), 1e-6f );
        EXPECT_EQ( elMesh.topology.org( e ), 6_v );
        EXPECT_EQ( elMesh.topology.dest( e ), 0_v );

        EXPECT_FALSE( elMesh.flipEdge( 28_e ) );

        EXPECT_TRUE( elMesh.flipEdge( e ) );
        EXPECT_NEAR( elMesh.edgeLengths[e], 1.f, 1e-6f );
        EXPECT_EQ( elMesh.topology.org( e ), 4_v );
        EXPECT_EQ( elMesh.topology.dest( e ), 7_v );
    }
}

} //namespace MR
