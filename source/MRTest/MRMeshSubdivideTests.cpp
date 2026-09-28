#include <MRMesh/MRMeshSubdivide.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMakePlane.h>
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

} //namespace MR
