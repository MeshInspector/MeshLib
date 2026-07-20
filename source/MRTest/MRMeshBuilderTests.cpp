#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRVertDuplication.h>
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
    // there are four edges with origin at vertex #0 having hole on one side (two with no left(e) and two with no right(e));
    // it is two open chains around vertex #0, and MeshBuilder has no issue with such configuration, so no duplication shall be done

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    ASSERT_EQ( duplicatedVerticesCnt, 0 );
    ASSERT_EQ( dups.size(), 0 );
    ASSERT_EQ( t[0_f], ( ThreeVertIds{ 0_v, 1_v, 2_v } ) );
    ASSERT_EQ( t[1_f], ( ThreeVertIds{ 0_v, 3_v, 4_v } ) );

    const auto topology = fromTrianglesDuplicatingNonManifoldVertices( t );
    EXPECT_EQ( topology.numValidVerts(), 5 );
    EXPECT_EQ( topology.numValidFaces(), 2 );
}

// duplication of one vertex can resolve non-manifoldness in its neighbor vertex, which shall not be duplicated then
TEST( MRMesh, duplicateResolvedNeighborVertex )
{
    Triangulation t;
    t.push_back( { 1_v, 2_v, 3_v } ); //0_f
    t.push_back( { 2_v, 1_v, 4_v } ); //1_f
    t.push_back( { 1_v, 2_v, 5_v } ); //2_f
    // the edge (1,2) is shared by three triangles, so both vertices 1 and 2 have repeated neighbor vertices;
    // duplication of vertex 1 in one of the triangles resolves vertex 2 into two open chains requiring nothing

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    EXPECT_EQ( duplicatedVerticesCnt, 1 );
    ASSERT_EQ( dups.size(), 1 );
    EXPECT_EQ( dups[0].srcVert, 1_v );
    EXPECT_EQ( dups[0].dupVert, 6_v );

    const auto topology = fromTrianglesDuplicatingNonManifoldVertices( t );
    EXPECT_EQ( topology.numValidVerts(), 6 );
    EXPECT_EQ( topology.numValidFaces(), 3 );
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

// a vertex with two closed rings of triangles around it, from a real mesh
TEST( MRMesh, inspectVertNeighbourhood )
{
    Triangulation t;
    t.push_back( { 992_v, 991_v, 990_v } );     //0_f
    t.push_back( { 992_v, 988_v, 1911_v } );    //1_f
    t.push_back( { 992_v, 989_v, 988_v } );     //2_f
    t.push_back( { 992_v, 1911_v, 989_v } );    //3_f
    t.push_back( { 992_v, 1020_v, 1019_v } );   //4_f
    t.push_back( { 992_v, 990_v, 1020_v } );    //5_f
    t.push_back( { 992_v, 1019_v, 460079_v } ); //6_f
    t.push_back( { 992_v, 1013_v, 991_v } );    //7_f
    t.push_back( { 992_v, 460079_v, 1013_v } ); //8_f

    std::vector<VertTri> recs;
    for ( FaceId f = 0_f; f < t.size(); ++f )
        recs.push_back( { 992_v, f } );

    // 6-triangle closed ring (991,990,1020,1019,460079,1013) and 3-triangle closed ring (988,1911,989)
    const auto info = inspectVertNeighbourhood( t, recs.data(), recs.data() + recs.size() );
    EXPECT_FALSE( info.hasRepeatedVerts() );
    EXPECT_EQ( info.numOpenChains(), 0u );
    EXPECT_EQ( info.numClosedChains(), 2u );

    // the vertex is non-manifold, so one of its rings must get a duplicate
    std::vector<VertDuplication> dups;
    EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups ), 1 );
    ASSERT_EQ( dups.size(), 1 );
    EXPECT_EQ( dups[0].srcVert, 992_v );
}

// check that VertInfo counters saturate at their maximum values instead of overflowing
TEST( MRMesh, inspectVertNeighbourhoodSaturation )
{
    // more disjoint triangles around one vertex than numOpenChains can store
    {
        Triangulation t;
        std::vector<VertTri> recs;
        const int n = int( VertInfo::maxNumOpenChains ) + 100;
        for ( int i = 0; i < n; ++i )
        {
            t.push_back( { 0_v, VertId( 1 + 2 * i ), VertId( 2 + 2 * i ) } );
            recs.push_back( { 0_v, FaceId( i ) } );
        }
        const auto info = inspectVertNeighbourhood( t, recs.data(), recs.data() + recs.size() );
        EXPECT_FALSE( info.hasRepeatedVerts() );
        EXPECT_EQ( info.numOpenChains(), VertInfo::maxNumOpenChains );
        EXPECT_EQ( info.numClosedChains(), 0u );
    }

    // more closed rings around one vertex than numClosedChains can store
    {
        Triangulation t;
        std::vector<VertTri> recs;
        const int n = int( VertInfo::maxNumClosedChains ) + 100;
        for ( int i = 0; i < n; ++i )
        {
            const VertId a( 1 + 2 * i ), b( 2 + 2 * i );
            t.push_back( { 0_v, a, b } );
            t.push_back( { 0_v, b, a } );
            recs.push_back( { 0_v, FaceId( 2 * i ) } );
            recs.push_back( { 0_v, FaceId( 2 * i + 1 ) } );
        }
        const auto info = inspectVertNeighbourhood( t, recs.data(), recs.data() + recs.size() );
        EXPECT_FALSE( info.hasRepeatedVerts() );
        EXPECT_EQ( info.numOpenChains(), 0u );
        EXPECT_EQ( info.numClosedChains(), VertInfo::maxNumClosedChains );
    }

    // the same triangle repeated many times: repetitions are counted, chain counters read as zero
    {
        Triangulation t;
        std::vector<VertTri> recs;
        const int n = 100;
        for ( int i = 0; i < n; ++i )
        {
            t.push_back( { 0_v, 1_v, 2_v } );
            recs.push_back( { 0_v, FaceId( i ) } );
        }
        const auto info = inspectVertNeighbourhood( t, recs.data(), recs.data() + recs.size() );
        EXPECT_TRUE( info.hasRepeatedVerts() );
        EXPECT_EQ( info.numRepeatedVerts(), 2u * ( n - 1 ) );
        EXPECT_EQ( info.numOpenChains(), 0u );
        EXPECT_EQ( info.numClosedChains(), 0u );
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

TEST( MRMesh, computeTrianglesRepetitions )
{
    Triangulation t;
    EXPECT_TRUE( computeTrianglesRepetitions( t ).empty() );

    t.push_back( { 0_v, 1_v, 2_v } ); //0_f
    t.push_back( { 0_v, 2_v, 3_v } ); //1_f
    t.push_back( { 2_v, 0_v, 3_v } ); //2_f same as 1_f with the opposite orientation
    t.push_back( { 1_v, 2_v, 0_v } ); //3_f same as 0_f
    t.push_back( { 2_v, 0_v, 1_v } ); //4_f same as 0_f
    t.push_back( { 3_v, 4_v, 5_v } ); //5_f

    const auto reps = computeTrianglesRepetitions( t );
    ASSERT_EQ( reps.size(), 3 );
    EXPECT_EQ( reps[0], 1 ); // {3,4,5}
    EXPECT_EQ( reps[1], 1 ); // {0,2,3}
    EXPECT_EQ( reps[2], 1 ); // {0,1,2}
}

} //namespace MeshBuilder

} //namespace MR
