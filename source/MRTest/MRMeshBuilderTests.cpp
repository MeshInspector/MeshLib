#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMeshBuilderTypes.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>

namespace MR
{

namespace MeshBuilder
{

// check non-manifold vertices resolving
TEST( MRMesh, duplicateNonManifoldVertices )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } ); //0_f
    t.push_back( { 0_v, 2_v, 3_v } ); //1_f
    t.push_back( { 0_v, 3_v, 1_v } ); //2_f

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 0 );
    ASSERT_EQ( dups.size(), 0 );

    t.push_back( { 0_v, 4_v, 5_v } ); //3_f
    t.push_back( { 0_v, 5_v, 6_v } ); //4_f
    t.push_back( { 0_v, 6_v, 4_v } ); //5_f

    duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 1 );
    ASSERT_EQ( dups.size(), 1 );
    ASSERT_EQ( dups[0].srcVert, 0 );
    ASSERT_EQ( dups[0].dupVert, 7 );

    int firstChangedTriangleNum = t[0_f][0] != 0 ? 0 : 3;
    for ( FaceId i{ firstChangedTriangleNum }; i < firstChangedTriangleNum + 3; ++i )
        ASSERT_EQ( t[i][0], 7 );
}

static void testBuildWithDups( const char * objMesh, int numVerts, int numComps )
{
    std::istringstream s( objMesh );
    auto maybeMesh = MeshLoad::fromObj( s );
    EXPECT_TRUE( maybeMesh );

    EXPECT_EQ( maybeMesh->topology.numValidVerts(), numVerts );
    EXPECT_EQ( MeshComponents::getNumComponents( *maybeMesh ), numComps );
}

TEST( MRMesh, MeshBuildWithDups )
{
    // this test case passed always
    testBuildWithDups
    (
        "v 0 0.5 0\n"
        "v -0.5 0.5 -0.5\n"
        "v 0.5 0.5 -0.5\n"
        "v -0.5 0.5 0.5\n"
        "v 0.5 0.5 0.5\n"
        "f 1 2 4\n"
        "f 1 4 2\n"
        "f 2 3 1\n"
        "f 1 5 4\n"
        "f 3 5 1\n"
        "f 2 1 3\n"
        "f 3 1 5\n"
        "f 1 4 5\n", 6, 1
    );

    // this test start passing only after recent improvements (but will fail in case vertex reordering)
    testBuildWithDups
    (
        "v 0 0.5 0\n"
        "v -0.5 0.5 -0.5\n"
        "v 0.5 0.5 -0.5\n"
        "v -0.5 0.5 0.5\n"
        "v 0.5 0.5 0.5\n"
        "f 1 2 4\n"
        "f 2 1 3\n"
        "f 1 4 5\n"
        "f 3 1 5\n"
        "f 1 4 2\n"
        "f 2 3 1\n"
        "f 1 5 4\n"
        "f 3 5 1\n", 6, 1
    );
}

} //namespace MeshBuilder

} //namespace MR
