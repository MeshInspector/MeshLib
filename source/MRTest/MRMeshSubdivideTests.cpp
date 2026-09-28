#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMakePlane.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MREdgeIterator.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, SubdivideMesh)
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    FaceBitSet region( 2 );
    region.set( 0_f );

    SubdivideSettings settings;
    settings.maxEdgeLen = 0.3f;
    settings.maxEdgeSplits = 1000;
    settings.maxDeviationAfterFlip = FLT_MAX;
    settings.region = &region;
    int splitsDone = subdivideMesh( mesh, settings );
    EXPECT_TRUE( splitsDone > 19 && splitsDone < 25 );
    EXPECT_TRUE( region.count() * 2 + 3 > mesh.topology.numValidFaces() );
    EXPECT_TRUE( region.count() * 2 - 3 > mesh.topology.numValidFaces() );

    settings.maxEdgeLen = 0.1f;
    settings.maxEdgeSplits = 10;
    splitsDone = subdivideMesh( mesh, settings );
    EXPECT_TRUE( splitsDone == 10 );
    EXPECT_TRUE( region.count() * 2 + 3 > mesh.topology.numValidFaces() );
    EXPECT_TRUE( region.count() * 2 - 3 > mesh.topology.numValidFaces() );
}

TEST(MRMesh, SubdivideMeshOnlyNearNotFlippable)
{
    Mesh base = makePlane();
    subdivideMesh( base, { .maxEdgeLen = 0.2f, .maxEdgeSplits = 10000 } );

    SubdivideSettings settings;
    settings.maxEdgeLen = 0.02f;
    settings.maxEdgeSplits = 100000;

    Mesh mesh = base;
    const int allSplits = subdivideMesh( mesh, settings );

    mesh = base;
    UndirectedEdgeBitSet notFlippable( mesh.topology.undirectedEdgeSize() );
    notFlippable.set( mesh.topology.edgeWithOrg( 0_v ).undirected() );
    settings.notFlippable = &notFlippable;
    settings.onlyNearNotFlippable = true;
    const int nearSplits = subdivideMesh( mesh, settings );
    EXPECT_GT( nearSplits, 0 );
    EXPECT_LT( 4 * nearSplits, allSplits );
    EXPECT_GT( notFlippable.count(), 1 );

    // without notFlippable nothing is split
    mesh = base;
    settings.notFlippable = nullptr;
    EXPECT_EQ( subdivideMesh( mesh, settings ), 0 );
}

// a split of notFlippable edge followed by a cascade of flips makes some edges near notFlippable
// only via the new vertex, so they must be queued as the edges opposite to it
TEST(MRMesh, SubdivideMeshOnlyNearNotFlippableOpposite)
{
    Mesh mesh;
    mesh.points = {
        { -0.5f, -0.5f, 0.f },
        { -0.5f, 0.5f, 0.f },
        { 0.5f, 0.5f, 0.f },
        { 0.5f, -0.5f, 0.f },
        { -0.11f, -0.107f, 0.002f },
        { 0.5f, 0.f, 0.f },
        { 0.f, -0.5f, 0.f },
        { -0.5f, 0.f, 0.f },
        { 0.f, 0.5f, 0.f },
        { 0.296f, -0.325f, 0.007f },
        { -0.237f, 0.367f, 0.02f },
        { -0.28f, -0.349f, -0.035f },
        { 0.302f, 0.238f, -0.004f },
        { 0.048f, 0.291f, -0.008f },
        { 0.25f, 0.5f, 0.f },
        { -0.153f, 0.032f, -0.002f },
        { -0.5f, 0.25f, 0.f },
        { -0.107f, -0.345f, 0.019f },
        { -0.25f, -0.5f, 0.f },
        { 0.151f, -0.051f, -0.023f },
        { 0.5f, -0.25f, 0.f },
        { 0.5f, 0.25f, 0.f },
        { 0.25f, -0.5f, 0.f },
        { -0.5f, -0.25f, 0.f },
        { -0.25f, 0.5f, 0.f }
    };
    Triangulation t = {
        { 24_v, 1_v, 10_v }, { 21_v, 2_v, 12_v }, { 4_v, 15_v, 11_v }, { 22_v, 3_v, 9_v },
        { 20_v, 5_v, 9_v }, { 17_v, 4_v, 11_v }, { 16_v, 7_v, 10_v }, { 14_v, 8_v, 12_v },
        { 19_v, 4_v, 9_v }, { 4_v, 17_v, 9_v }, { 15_v, 4_v, 10_v }, { 4_v, 13_v, 10_v },
        { 23_v, 0_v, 11_v }, { 18_v, 6_v, 11_v }, { 13_v, 4_v, 12_v }, { 4_v, 19_v, 12_v },
        { 13_v, 12_v, 8_v }, { 10_v, 13_v, 8_v }, { 14_v, 12_v, 2_v }, { 15_v, 10_v, 7_v },
        { 11_v, 15_v, 7_v }, { 16_v, 10_v, 1_v }, { 17_v, 11_v, 6_v }, { 9_v, 17_v, 6_v },
        { 18_v, 11_v, 0_v }, { 19_v, 9_v, 5_v }, { 12_v, 19_v, 5_v }, { 20_v, 9_v, 3_v },
        { 21_v, 12_v, 5_v }, { 22_v, 9_v, 6_v }, { 23_v, 11_v, 7_v }, { 24_v, 10_v, 8_v }
    };
    mesh.topology = MeshBuilder::fromTriangles( t );
    const auto nfEdge = mesh.topology.findEdge( 15_v, 10_v );
    ASSERT_TRUE( nfEdge );
    UndirectedEdgeBitSet notFlippable( mesh.topology.undirectedEdgeSize() );
    notFlippable.set( nfEdge.undirected() );

    SubdivideSettings settings;
    settings.maxEdgeLen = 0.05f;
    settings.maxEdgeSplits = 100000;
    settings.maxDeviationAfterFlip = FLT_MAX;
    settings.notFlippable = &notFlippable;
    settings.onlyNearNotFlippable = true;
    EXPECT_GT( subdivideMesh( mesh, settings ), 0 );

    // no long edge remains having a vertex of its left or right triangle incident to notFlippable edge
    const auto ends = getIncidentVerts( mesh.topology, notFlippable );
    const auto & topology = mesh.topology;
    for ( auto ue : undirectedEdges( topology ) )
    {
        const EdgeId e( ue );
        const bool near = ends.test( topology.org( e ) ) || ends.test( topology.dest( e ) )
            || ( topology.left( e ) && ends.test( topology.dest( topology.next( e ) ) ) )
            || ( topology.right( e ) && ends.test( topology.dest( topology.prev( e ) ) ) );
        if ( near )
            EXPECT_LT( mesh.edgeLength( e ), settings.maxEdgeLen );
    }
}

} //namespace MR
