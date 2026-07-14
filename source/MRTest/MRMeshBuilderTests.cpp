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

TEST( MRMesh, duplicateDoubleHoleVertex )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } ); //0_f
    t.push_back( { 0_v, 3_v, 4_v } ); //1_f
    // there are four edges with origin at vertex #0 having hole on one side (two with no left(e) and two with no right(e))

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 1 );
    ASSERT_EQ( dups.size(), 1 );
    ASSERT_EQ( dups[0].srcVert, 0_v );
    ASSERT_EQ( dups[0].dupVert, 5_v );
    ASSERT_EQ( t[0_f], ( ThreeVertIds{ 0_v, 1_v, 2_v } ) );
    ASSERT_EQ( t[1_f], ( ThreeVertIds{ 5_v, 3_v, 4_v } ) );
}

// check a vertex with both a closed ring of triangles and a separate chain around it
TEST( MRMesh, duplicateClosedPlusOpenChainVertex )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } ); //0_f
    t.push_back( { 0_v, 2_v, 3_v } ); //1_f
    t.push_back( { 0_v, 3_v, 1_v } ); //2_f closed ring around #0
    t.push_back( { 0_v, 4_v, 5_v } ); //3_f separate triangle sharing #0 only

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 1 );
    ASSERT_EQ( dups.size(), 1 );
    ASSERT_EQ( dups[0].srcVert, 0_v );
    ASSERT_EQ( dups[0].dupVert, 6_v );
    if ( t[3_f][0] == 0_v )
    {
        // the single triangle was walked first, so the closed ring got the duplicated vertex
        for ( FaceId i{ 0 }; i < 3; ++i )
            ASSERT_EQ( t[i][0], 6_v );
    }
    else
    {
        for ( FaceId i{ 0 }; i < 3; ++i )
            ASSERT_EQ( t[i][0], 0_v );
        ASSERT_EQ( t[3_f][0], 6_v );
    }
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
    // first 4 triangles subdivide a square with center point,
    // following 4 triangles subdivide the opposite side of same triangle,
    // all 5 points are shared,
    // expected that only center vetrex is duplicated and single connected component remains
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

    // same situation as above with the order of triangles changed
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

    // first 4 triangles subdivide a square with center point,
    // following 4 triangles subdivide same square with center point,
    // 3 points on one diagonal are shared,
    // it is divided properly on two components if the rings are computed from the smallest by id next triangle
    testBuildWithDups
    (
        "v -1 0 -1\n"
        "v  1 0 -1\n"
        "v -1 0  1\n"
        "v  1 0  1\n"
        "v -1 0 -1\n"
        "v  1 0  1\n"
        "v  0 0  0\n"
        "f 7 1 3\n"
        "f 1 7 2\n"
        "f 7 3 4\n"
        "f 2 7 4\n"
        "f 5 7 2\n"
        "f 7 3 6\n"
        "f 2 7 6\n"
        "f 7 5 3\n", 10, 2
    );
}

} //namespace MeshBuilder

} //namespace MR
