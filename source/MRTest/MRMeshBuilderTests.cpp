#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRVertDuplication.h>
#include <MRMesh/MRMeshBuilderTypes.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRUnorientedTriangle.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRRingIterator.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>
#include <tbb/global_control.h>
#include <algorithm>
#include <numeric>
#include <random>

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

// the vertices of an edge shared by three triangles are both duplicated, fully detaching the third triangle
TEST( MRMesh, duplicateTripleEdgeVertices )
{
    Triangulation t;
    t.push_back( { 1_v, 2_v, 3_v } ); //0_f
    t.push_back( { 2_v, 1_v, 4_v } ); //1_f
    t.push_back( { 1_v, 2_v, 5_v } ); //2_f
    // duplication of vertex 1 formally resolves vertex 2 into two open chains,
    // but vertex 2 is still duplicated to avoid two neighbor vertices with equal coordinates

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    EXPECT_EQ( duplicatedVerticesCnt, 2 );
    ASSERT_EQ( dups.size(), 2 );
    EXPECT_EQ( dups[0].srcVert, 1_v );
    EXPECT_EQ( dups[0].dupVert, 6_v );
    EXPECT_EQ( dups[1].srcVert, 2_v );
    EXPECT_EQ( dups[1].dupVert, 7_v );

    // the last triangle is fully detached from the first two
    EXPECT_EQ( t[2_f], ( ThreeVertIds{ 6_v, 7_v, 5_v } ) );

    const auto topology = fromTrianglesDuplicatingNonManifoldVertices( t );
    EXPECT_EQ( topology.numValidVerts(), 7 );
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

// a closed ring around #0 plus one opposite-orientation triangle sharing the ring's rim pair (2,3):
// the lone triangle 4_f is fully detached: all its three vertices are duplicated
TEST( MRMesh, duplicateVertexOppositeOrientedTri )
{
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } ); //0_f closed ring 1-2-3-4-1
    t.push_back( { 0_v, 2_v, 3_v } ); //1_f
    t.push_back( { 0_v, 3_v, 4_v } ); //2_f
    t.push_back( { 0_v, 4_v, 1_v } ); //3_f
    t.push_back( { 0_v, 3_v, 2_v } ); //4_f reversed rim pair of 1_f

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    EXPECT_EQ( duplicatedVerticesCnt, 3 );
    ASSERT_EQ( dups.size(), 3 );
    EXPECT_EQ( dups[0].srcVert, 0_v );
    EXPECT_EQ( dups[0].dupVert, 5_v );
    EXPECT_EQ( dups[1].srcVert, 2_v );
    EXPECT_EQ( dups[1].dupVert, 6_v );
    EXPECT_EQ( dups[2].srcVert, 3_v );
    EXPECT_EQ( dups[2].dupVert, 7_v );

    // the ring is intact and the lone triangle is fully detached
    for ( FaceId i{ 0 }; i < 4; ++i )
        EXPECT_EQ( t[i][0], 0_v );
    EXPECT_EQ( t[4_f], ( ThreeVertIds{ 5_v, 7_v, 6_v } ) );

    // the result is manifold
    dups.clear();
    EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups ), 0 );
    EXPECT_EQ( dups.size(), 0 );
}

// a fan around #0 with two continuation options from the shared rim vertex #2;
// the given predicate prefers the triangle with the remaining vertex #4,
// so the other continuation 1_f is fully detached; without the predicate the first option by face id wins instead
TEST( MRMesh, duplicateVertexPreferredContinuation )
{
    Triangulation initT;
    initT.push_back( { 0_v, 1_v, 2_v } ); //0_f
    initT.push_back( { 0_v, 2_v, 3_v } ); //1_f
    initT.push_back( { 0_v, 2_v, 4_v } ); //2_f

    {
        auto t = initT;
        auto prefer4 = []( VertId e0, VertId e1, VertId vRef, VertId vCand, VertId vBest )
        {
            if ( e1 == 0_v )
            {
                // the branching during the walk around #0: the reference triangle is (2, 0, 1) = 0_f up to rotation,
                // and the candidates are (0, 2, 3) = 1_f and (0, 2, 4) = 2_f
                EXPECT_EQ( e0, 2_v );
                EXPECT_EQ( vRef, 1_v );
                EXPECT_EQ( vBest, 3_v );
            }
            return vCand == 4_v;
        };
        std::vector<VertDuplication> dups;
        EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups, {}, prefer4 ), 2 );
        ASSERT_EQ( dups.size(), 2 );
        EXPECT_EQ( dups[0].srcVert, 0_v );
        EXPECT_EQ( dups[0].dupVert, 5_v );
        EXPECT_EQ( dups[1].srcVert, 2_v );
        EXPECT_EQ( dups[1].dupVert, 6_v );
        EXPECT_EQ( t[0_f], ( ThreeVertIds{ 0_v, 1_v, 2_v } ) );
        EXPECT_EQ( t[1_f], ( ThreeVertIds{ 5_v, 6_v, 3_v } ) );
        EXPECT_EQ( t[2_f], ( ThreeVertIds{ 0_v, 2_v, 4_v } ) );

        // the result is manifold
        dups.clear();
        EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups, {}, prefer4 ), 0 );
    }
    {
        auto t = initT;
        std::vector<VertDuplication> dups;
        EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups ), 2 );
        EXPECT_EQ( t[1_f], ( ThreeVertIds{ 0_v, 2_v, 3_v } ) );
        EXPECT_EQ( t[2_f], ( ThreeVertIds{ 5_v, 6_v, 4_v } ) );
    }
}

// a hub vertex #3 with two open chains (1-2-0, 6-7-8) and a closed ring (0-4-5) sharing rim vertex #0;
// duplicating #0 opens the ring into one long chain around the hub,
// and the chain 6-7-8 is split off with its own duplicate of the hub
TEST( MRMesh, duplicateVertexWithThreeChains )
{
    Triangulation t;
    t.push_back( { 3_v, 1_v, 2_v } ); //0_f open chain 1-2-0
    t.push_back( { 3_v, 2_v, 0_v } ); //1_f
    t.push_back( { 3_v, 0_v, 4_v } ); //2_f closed ring 0-4-5-0
    t.push_back( { 3_v, 4_v, 5_v } ); //3_f
    t.push_back( { 3_v, 5_v, 0_v } ); //4_f
    t.push_back( { 3_v, 6_v, 7_v } ); //5_f open chain 6-7-8
    t.push_back( { 3_v, 7_v, 8_v } ); //6_f

    std::vector<VertDuplication> dups;
    size_t duplicatedVerticesCnt = duplicateNonManifoldVertices( t, nullptr, &dups );
    EXPECT_EQ( duplicatedVerticesCnt, 2 );
    ASSERT_EQ( dups.size(), 2 );
    EXPECT_EQ( dups[0].srcVert, 0_v );
    EXPECT_EQ( dups[0].dupVert, 9_v );
    EXPECT_EQ( dups[1].srcVert, 3_v );
    EXPECT_EQ( dups[1].dupVert, 10_v );

    // the first chain and the opened ring keep the hub vertex #3, the second chain gets its duplicate #10
    EXPECT_EQ( t[0_f][0], 3_v );
    EXPECT_EQ( t[1_f][0], 3_v );
    EXPECT_EQ( t[2_f][0], 3_v );
    EXPECT_EQ( t[3_f][0], 3_v );
    EXPECT_EQ( t[4_f], ( ThreeVertIds{ 3_v, 5_v, 9_v } ) );
    EXPECT_EQ( t[5_f][0], 10_v );
    EXPECT_EQ( t[6_f][0], 10_v );

    // the result is manifold
    dups.clear();
    EXPECT_EQ( duplicateNonManifoldVertices( t, nullptr, &dups ), 0 );
    EXPECT_EQ( dups.size(), 0 );
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

static void testBuildWithDups( const char * objMesh, int numFaces, int numVerts, int numComps )
{
    std::istringstream s( objMesh );
    auto maybeMesh = MeshLoad::fromObj( s );
    EXPECT_TRUE( maybeMesh );

    EXPECT_EQ( maybeMesh->topology.numValidFaces(), numFaces );
    EXPECT_EQ( maybeMesh->topology.numValidVerts(), numVerts );
    EXPECT_EQ( MeshComponents::getNumComponents( *maybeMesh ), numComps );
}

TEST( MRMesh, MeshBuildWithDups )
{
    // first 4 triangles subdivide a square with center point,
    // following 4 triangles subdivide the opposite side of same triangle,
    // all 5 points are shared,
    // the two identical oppositely oriented sides are fully separated in two components
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
        "f 1 4 5\n", 8, 10, 2
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
        "f 3 5 1\n", 8, 10, 2
    );

    // first 4 triangles subdivide a square with center point,
    // following 4 triangles subdivide same square with center point,
    // 3 points on one diagonal are shared,
    // the two squares are fully separated in two components
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
        "f 7 5 3\n", 8, 10, 2
    );

    // two cubes with mirrored triangulations sharing all 8 vertices;
    // every vertex must be duplicated exactly once giving two separate cube components
    testBuildWithDups
    (
        "v -0.5 -0.5 -0.5\n"
        "v -0.5 0.5 -0.5\n"
        "v 0.5 0.5 -0.5\n"
        "v 0.5 -0.5 -0.5\n"
        "v -0.5 -0.5 0.5\n"
        "v -0.5 0.5 0.5\n"
        "v 0.5 0.5 0.5\n"
        "v 0.5 -0.5 0.5\n"
        "f 7 5 6\n"
        "f 5 7 8\n"
        "f 2 7 6\n"
        "f 7 2 3\n"
        "f 7 4 8\n"
        "f 4 7 3\n"
        "f 1 2 3\n"
        "f 3 4 1\n"
        "f 1 5 6\n"
        "f 6 2 1\n"
        "f 1 4 8\n"
        "f 8 5 1\n"
        "f 7 6 5\n"
        "f 5 8 7\n"
        "f 2 6 7\n"
        "f 7 3 2\n"
        "f 7 8 4\n"
        "f 4 3 7\n"
        "f 1 3 2\n"
        "f 3 1 4\n"
        "f 1 6 5\n"
        "f 6 1 2\n"
        "f 1 8 4\n"
        "f 8 1 5\n", 24, 16, 2
    );

    // a soup of mirrored and repeated triangles over 6 vertices mixing twin chains and neighbourhoods
    // with unevenly repeated vertices, sensitive to the processing order and chain construction rules
    testBuildWithDups
    (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 1 0\n"
        "v 0 0 1\n"
        "v 1 1 0\n"
        "v 1 0 1\n"
        "f 3 2 5\n"
        "f 3 5 6\n"
        "f 4 1 6\n"
        "f 4 2 3\n"
        "f 3 6 4\n"
        "f 2 4 1\n"
        "f 5 2 3\n"
        "f 4 6 5\n"
        "f 3 4 5\n"
        "f 1 3 6\n"
        "f 6 2 4\n", 11, 13, 2
    );
}

// triangles with invalid vertex ids are neither added nor lost
TEST( MRMesh, fromTrianglesInvalidVerts )
{
    for ( int n : { 20, 150 } ) // 2*(n-1)^2 triangles: sequential and parallel builds
    {
        Triangulation t;
        for ( int y = 0; y + 1 < n; ++y )
            for ( int x = 0; x + 1 < n; ++x )
            {
                const int v = y * n + x;
                t.push_back( { VertId( v ), VertId( v + 1 ), VertId( v + n ) } );
                t.push_back( { VertId( v + 1 ), VertId( v + n + 1 ), VertId( v + n ) } );
            }
        const int numGood = (int)t.size();
        t.push_back( { VertId(), VertId(), VertId() } );
        t.push_back( { VertId( -2 ), VertId( -3 ), VertId( -4 ) } );
        t.push_back( { VertId(), 0_v, 1_v } );
        t.push_back( { VertId(), VertId( n * n ), VertId( n * n + 1 ) } );

        FaceBitSet region( t.size(), true );
        int skipped = 0;
        const auto topology = fromTriangles( t, { .region = &region, .skippedFaceCount = &skipped } );
        EXPECT_EQ( topology.numValidFaces(), numGood );
        EXPECT_EQ( skipped, 4 );
        EXPECT_EQ( region.count(), 4 );
        EXPECT_EQ( region.find_first(), FaceId( numGood ) );
    }
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

// the same faces and the same counter-clockwise ring of neighbours around every vertex
static void expectSameConnectivity( const MeshTopology & a, const MeshTopology & b )
{
    EXPECT_TRUE( b.checkValidity() );
    EXPECT_EQ( a.edgeSize(), b.edgeSize() );
    EXPECT_EQ( a.getValidVerts(), b.getValidVerts() );
    ASSERT_EQ( a.getValidFaces(), b.getValidFaces() );
    for ( FaceId f : a.getValidFaces() )
    {
        auto va = a.getTriVerts( f );
        auto vb = b.getTriVerts( f );
        std::rotate( vb.begin(), std::find( vb.begin(), vb.end(), va[0] ), vb.end() );
        EXPECT_EQ( va, vb );
    }
    std::vector<VertId> ra, rb;
    for ( VertId v : a.getValidVerts() )
    {
        ra.clear();
        rb.clear();
        for ( EdgeId e : orgRing( a, v ) )
            ra.push_back( a.dest( e ) );
        for ( EdgeId e : orgRing( b, v ) )
            rb.push_back( b.dest( e ) );
        ASSERT_EQ( ra.size(), rb.size() );
        std::rotate( rb.begin(), std::find( rb.begin(), rb.end(), ra[0] ), rb.end() );
        EXPECT_EQ( ra, rb );
    }
}

TEST( MRMesh, buildFromManifoldTriangles )
{
    // closed mesh with shuffled vertices and triangles, as built sequentially triangle by triangle
    const auto sphere = makeSphere( { .numMeshVertices = 3000 } );
    Triangulation t = sphere.topology.getTriangulation();
    std::mt19937 rng( 1 );
    // the same on all platforms, unlike std::shuffle
    auto shuffle = [&]( auto & v, auto id )
    {
        for ( size_t i = v.size(); i > 1; --i )
            std::swap( v[decltype( id )( i - 1 )], v[decltype( id )( rng() % i )] );
    };
    std::vector<int> perm( sphere.topology.vertSize() );
    std::iota( perm.begin(), perm.end(), 0 );
    shuffle( perm, size_t{} );
    for ( auto & tri : t )
        for ( auto & v : tri )
            v = VertId( perm[v] );
    shuffle( t, FaceId{} );
    // remove some triangles to get holes
    for ( int i = 0; i < 30; ++i )
        t.erase( t.begin() + int( rng() % t.size() ) );

    FaceBitSet region( t.size(), true );
    const auto seq = fromTriangles( t, { .region = &region } );
    EXPECT_TRUE( region.none() );
    MeshTopology direct;
    int skipped = -1;
    EXPECT_TRUE( direct.buildFromManifoldTriangles( t, { .skippedFaceCount = &skipped } ) );
    EXPECT_EQ( skipped, 0 );
    expectSameConnectivity( seq, direct );

    MeshTopology oneThread;
    {
        tbb::global_control singleThread( tbb::global_control::max_allowed_parallelism, 1 );
        EXPECT_TRUE( oneThread.buildFromManifoldTriangles( t ) );
    }
    EXPECT_EQ( oneThread, direct );

    // triangles with repeated or invalid vertices are skipped
    t.push_back( { 0_v, 0_v, 1_v } );
    t.push_back( { VertId(), VertId(), VertId() } );
    EXPECT_TRUE( direct.buildFromManifoldTriangles( t, { .skippedFaceCount = &skipped } ) );
    EXPECT_EQ( skipped, 2 );
    EXPECT_EQ( direct.numValidFaces(), int( t.size() ) - 2 );
}

TEST( MRMesh, buildFromManifoldTrianglesRegion )
{
    const auto sphere = makeSphere( { .numMeshVertices = 3000 } );
    Triangulation t = sphere.topology.getTriangulation();
    t.push_back( { 0_v, 0_v, 1_v } );
    FaceBitSet region( t.size(), true );
    for ( FaceId f = 0_f; f < 100; ++f )
        region.reset( f );
    constexpr int shift = 10;

    FaceBitSet seqRegion = region, directRegion = region;
    int seqSkipped = -1, directSkipped = -1;
    MeshTopology seq;
    addTriangles( seq, t, { .region = &seqRegion, .shiftFaceId = shift, .skippedFaceCount = &seqSkipped } );
    MeshTopology direct;
    EXPECT_TRUE( direct.buildFromManifoldTriangles( t, { .region = &directRegion, .shiftFaceId = shift, .skippedFaceCount = &directSkipped } ) );
    EXPECT_EQ( seqSkipped, 1 );
    EXPECT_EQ( directSkipped, 1 );
    EXPECT_EQ( directRegion.count(), 1 );
    EXPECT_TRUE( directRegion.test( FaceId( int( t.size() ) - 1 ) ) );
    EXPECT_EQ( seqRegion.count(), 1 );
    EXPECT_EQ( direct.faceSize(), t.size() + shift );
    EXPECT_FALSE( direct.hasFace( 0_f ) );
    expectSameConnectivity( seq, direct );
}

TEST( MRMesh, duplicateNonManifoldVerticesBigMesh )
{
    // two closed spheres sharing one vertex, with enough triangles for fromTriangles to try buildFromManifoldTriangles
    const auto sphere = makeSphere( { .numMeshVertices = 10000 } );
    const Triangulation one = sphere.topology.getTriangulation();
    const int nv = int( sphere.topology.vertSize() );
    Triangulation t = one;
    for ( auto tri : one )
    {
        for ( auto & v : tri )
            v = v == 0_v ? v : VertId( v + nv );
        t.push_back( tri );
    }
    ASSERT_GE( t.size(), 8192 );

    MeshTopology topology;
    bool nonManifoldVertsOnly = false;
    EXPECT_FALSE( topology.buildFromManifoldTriangles( t, {}, &nonManifoldVertsOnly ) );
    EXPECT_TRUE( nonManifoldVertsOnly );

    std::vector<VertDuplication> dups;
    topology = fromTrianglesDuplicatingNonManifoldVertices( t, &dups );
    ASSERT_EQ( dups.size(), 1 );
    EXPECT_EQ( dups[0].srcVert, 0_v );
    EXPECT_EQ( topology.numValidFaces(), int( t.size() ) );
    EXPECT_EQ( topology.numValidVerts(), 2 * nv );
    EXPECT_TRUE( topology.checkValidity() );
    EXPECT_EQ( topology.findNumHoles(), 0 );
}

TEST( MRMesh, buildFromManifoldTrianglesFails )
{
    // two triangles with one vertex in common: two open fans there
    const Triangulation bowtie = { { 0_v, 1_v, 2_v }, { 0_v, 3_v, 4_v } };
    MeshTopology topology;
    EXPECT_TRUE( topology.buildFromManifoldTriangles( bowtie ) );
    EXPECT_TRUE( topology.checkValidity() );
    EXPECT_EQ( topology.findNumHoles(), fromTriangles( bowtie ).findNumHoles() );

    // three triangles on edge 0-1
    EXPECT_FALSE( topology.buildFromManifoldTriangles( { { 0_v, 1_v, 2_v }, { 1_v, 0_v, 3_v }, { 1_v, 0_v, 4_v } } ) );
    // two triangles with edge 0-1 in the same direction
    EXPECT_FALSE( topology.buildFromManifoldTriangles( { { 0_v, 1_v, 2_v }, { 1_v, 0_v, 3_v }, { 0_v, 1_v, 4_v } } ) );
    bool nonManifoldVertsOnly = true;
    EXPECT_FALSE( topology.buildFromManifoldTriangles( { { 0_v, 1_v, 2_v }, { 1_v, 0_v, 3_v }, { 0_v, 1_v, 4_v } }, {}, &nonManifoldVertsOnly ) );
    EXPECT_FALSE( nonManifoldVertsOnly );
    // two closed fans around vertex 0
    const Triangulation twoTetrahedra = {
        { 0_v, 1_v, 2_v }, { 0_v, 2_v, 3_v }, { 0_v, 3_v, 1_v }, { 1_v, 3_v, 2_v },
        { 0_v, 4_v, 5_v }, { 0_v, 5_v, 6_v }, { 0_v, 6_v, 4_v }, { 4_v, 6_v, 5_v } };
    EXPECT_FALSE( topology.buildFromManifoldTriangles( twoTetrahedra, {}, &nonManifoldVertsOnly ) );
    EXPECT_TRUE( nonManifoldVertsOnly );
    // still valid after the failures
    EXPECT_TRUE( topology.checkValidity() );
    EXPECT_EQ( topology.numValidFaces(), 2 );
}

} //namespace MeshBuilder

TEST( MRMesh, UnorientedTriangleHash )
{
    // all triangles with close vertex ids get distinct hash values
    for ( int first : { 0, 1'000'000'000 } )
    {
        constexpr int n = 64;
        std::vector<size_t> hashes;
        for ( int a = first; a < first + n; ++a )
            for ( int b = a + 1; b < first + n; ++b )
                for ( int c = b + 1; c < first + n; ++c )
                    hashes.push_back( std::hash<UnorientedTriangle>{}( UnorientedTriangle( { VertId( a ), VertId( b ), VertId( c ) } ) ) );
        std::sort( hashes.begin(), hashes.end() );
        EXPECT_EQ( std::adjacent_find( hashes.begin(), hashes.end() ), hashes.end() );
    }
}

} //namespace MR
