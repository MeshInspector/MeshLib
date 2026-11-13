#include <gtest/gtest.h>
#include <MRMesh/MRMeshDecimate.h>
#include <MRMesh/MRCylinder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRBuffer.h>
#include <MRMesh/MRMeshLoad.h>
#include <MRMesh/MRObjectMeshData.h>
#include <MRMesh/MRMeshProject.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRQuadraticForm.h>
#include <map>
#include <random>
#include <tuple>
#include <sstream>

namespace MR
{

// check if Decimator updates region
TEST( MRMesh, MeshDecimate )
{
    Mesh meshCylinder = makeCylinderAdvanced(0.5f, 0.5f, 0.0f, 20.0f / 180.0f * PI_F, 1.0f, 16);

    // select all faces
    MR::FaceBitSet regionForDecimation = meshCylinder.topology.getValidFaces();
    MR::FaceBitSet regionSaved(regionForDecimation);

    // setup and run decimator
    DecimateSettings decimateSettings;
    decimateSettings.maxError = 0.001f;
    decimateSettings.region = &regionForDecimation;
    decimateSettings.maxTriangleAspectRatio = 80.0f;

    auto decimateResults = decimateMesh(meshCylinder, decimateSettings);

    // compare regions and deleted vertices and faces
    ASSERT_NE(regionSaved, regionForDecimation);
    ASSERT_GT(decimateResults.vertsDeleted, 0);
    ASSERT_GT(decimateResults.facesDeleted, 0);

    ASSERT_TRUE(meshCylinder.topology.checkValidity());
    meshCylinder.packOptimally();
    ASSERT_TRUE(meshCylinder.topology.checkValidity());
}

TEST( MRMesh, MeshDecimateParallel )
{
    const int cNumVerts = 400;
    for ( bool preventNormalFlips : { true, false } )
    {
        auto mesh = makeSphere( { .numMeshVertices = cNumVerts } );
        mesh.packOptimally();
        DecimateSettings settings
        {
            .maxError = 1000000, // no actual limit
            .preventNormalFlips = preventNormalFlips,
            .maxDeletedVertices = cNumVerts - 3,
            .subdivideParts = 8
        };
        decimateMesh( mesh, settings );
        // the last tetrahedron is not collapsed in two triangles lying on each other, and its removal would exceed the limit
        EXPECT_EQ( mesh.topology.numValidFaces(), 4 ) << preventNormalFlips;
        EXPECT_EQ( mesh.topology.numValidVerts(), 4 ) << preventNormalFlips;

        // without the limit, the tetrahedron is removed completely
        decimateMesh( mesh, { .maxError = 1000000, .preventNormalFlips = preventNormalFlips } );
        EXPECT_EQ( mesh.topology.numValidFaces(), 0 ) << preventNormalFlips;
        EXPECT_EQ( mesh.topology.numValidVerts(), 0 ) << preventNormalFlips;
    }
}

// a closed tetrahedron is removed completely if its four vertices can be collapsed in one point within maxError, otherwise it is kept whole
TEST( MRMesh, MeshDecimateTetrahedron )
{
    std::mt19937 gen( 1 );
    auto rnd = [&] { return float( gen() % 2001 ) / 1000 - 1; };
    int removed = 0, keptWithSmallEdge = 0;
    for ( int i = 0; i < 200; ++i )
    {
        VertCoords pts;
        for ( int k = 0; k < 4; ++k )
        {
            const float x = rnd(), y = rnd(), z = rnd();
            pts.push_back( { x, y, z } );
        }
        if ( dot( pts[1_v] - pts[0_v], cross( pts[2_v] - pts[0_v], pts[3_v] - pts[0_v] ) ) < 0 )
            std::swap( pts[1_v], pts[2_v] );
        Mesh mesh = Mesh::fromTriangles( pts, { { 0_v, 2_v, 1_v }, { 0_v, 1_v, 3_v }, { 0_v, 3_v, 2_v }, { 1_v, 2_v, 3_v } } );
        const DecimateSettings settings{ .maxError = 0.03f * ( 1 + i % 10 ) };
        const auto forms = computeFormsAtVertices( mesh, settings.stabilizer, settings.angleWeightedDistToPlane );
        auto collapse = [&]( int a, int b ) { return sum( forms[VertId( a )], pts[VertId( a )], forms[VertId( b )], pts[VertId( b )] ); };
        const auto [q01, p01] = collapse( 0, 1 );
        const auto [q23, p23] = collapse( 2, 3 );
        const float errSq = sum( q01, p01, q23, p23 ).first.c, maxErrSq = sqr( settings.maxError );
        if ( std::abs( errSq - maxErrSq ) <= 1e-3f * maxErrSq )
            continue; // the decimator may sum the forms in another order, so the rounding can decide such case
        const bool small = errSq <= maxErrSq;
        bool smallEdge = false;
        for ( int a = 0; a < 4; ++a )
            for ( int b = a + 1; b < 4; ++b )
                smallEdge = smallEdge || collapse( a, b ).first.c <= sqr( settings.maxError );

        decimateMesh( mesh, settings );
        EXPECT_EQ( mesh.topology.numValidFaces(), small ? 0 : 4 ) << i;
        removed += small;
        keptWithSmallEdge += !small && smallEdge;
    }
    EXPECT_GT( removed, 0 );
    EXPECT_GT( keptWithSmallEdge, 0 ); // one of their edges alone could be collapsed within maxError
}

// a small tetrahedron is kept whole if the settings do not permit its removal
TEST( MRMesh, MeshDecimateTetrahedronKept )
{
    const Mesh tetra = Mesh::fromTriangles( { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } },
        { { 0_v, 2_v, 1_v }, { 0_v, 1_v, 3_v }, { 0_v, 3_v, 2_v }, { 1_v, 2_v, 3_v } } );
    const auto e12 = tetra.topology.findEdge( 1_v, 2_v ).undirected();
    auto facesLeft = [&]( const DecimateSettings & settings )
    {
        Mesh mesh = tetra;
        decimateMesh( mesh, settings );
        return mesh.topology.numValidFaces();
    };
    EXPECT_EQ( facesLeft( { .maxError = 10 } ), 0 );
    EXPECT_EQ( facesLeft( { .maxError = 10, .maxDeletedFaces = 3 } ), 4 );

    FaceBitSet region( 4, true );
    region.reset( 3_f );
    EXPECT_EQ( facesLeft( { .maxError = 10, .region = &region } ), 4 );

    UndirectedEdgeBitSet notFlippable( 6 );
    notFlippable.set( e12 );
    EXPECT_EQ( facesLeft( { .maxError = 10, .notFlippable = &notFlippable } ), 4 );

    UndirectedEdgeBitSet edgesToCollapse( 6, true );
    edgesToCollapse.reset( e12 );
    EXPECT_EQ( facesLeft( { .maxError = 10, .edgesToCollapse = &edgesToCollapse } ), 4 );

    VertBitSet bdVerts( 4 );
    bdVerts.set( 3_v );
    EXPECT_EQ( facesLeft( { .maxError = 10, .touchBdVerts = false, .bdVerts = &bdVerts } ), 4 );

    EXPECT_EQ( facesLeft( { .maxError = 10, .preCollapse = []( EdgeId, const Vector3f & ) { return false; } } ), 4 );

    // preCollapse is called only right before the removal, since it may update vertex attributes
    int preCollapseCalls = 0;
    auto countCalls = [&]( EdgeId, const Vector3f & ) { ++preCollapseCalls; return true; };
    EXPECT_EQ( facesLeft( { .maxError = 10, .maxDeletedFaces = 3, .preCollapse = countCalls } ), 4 );
    EXPECT_EQ( preCollapseCalls, 0 );
    EXPECT_EQ( facesLeft( { .maxError = 10, .preCollapse = countCalls } ), 0 );
    EXPECT_EQ( preCollapseCalls, 1 );

    // for ShortestEdgeFirst, maxError limits the length of the edges, and the longest one is sqrt(2)
    EXPECT_EQ( facesLeft( { .strategy = DecimateStrategy::ShortestEdgeFirst, .maxError = 1.2f } ), 4 );
    EXPECT_EQ( facesLeft( { .strategy = DecimateStrategy::ShortestEdgeFirst, .maxError = 1.5f } ), 0 );
}

// all parts modes decimate a region of the mesh in parallel and keep the sets from the settings consistent with the mesh
TEST( MRMesh, MeshDecimatePartsModes )
{
    for ( auto mode : { DecimatePartsMode::PackedMesh, DecimatePartsMode::AABBTree } )
    {
        auto mesh = makeSphere( { .numMeshVertices = 2000 } );
        FaceBitSet region( mesh.topology.faceSize() );
        for ( auto f : mesh.topology.getValidFaces() )
            if ( mesh.triCenter( f ).z > 0 )
                region.set( f );
        const auto numRegionFaces = int( region.count() );
        auto notFlippable = findRegionBoundaryUndirectedEdgesInsideMesh( mesh.topology, region );
        Vector<QuadraticForm3f, VertId> vertForms;

        DecimateSettings settings;
        settings.maxError = FLT_MAX;
        settings.maxDeletedFaces = numRegionFaces / 2;
        settings.region = &region;
        settings.notFlippable = &notFlippable;
        settings.vertForms = &vertForms;
        settings.subdivideParts = 8;
        settings.partsMode = mode;
        const auto res = decimateMesh( mesh, settings );

        EXPECT_GE( res.facesDeleted, numRegionFaces / 2 );
        EXPECT_LE( res.facesDeleted, numRegionFaces / 2 + 1 );
        EXPECT_TRUE( mesh.topology.checkValidity() );
        EXPECT_LT( int( region.count() ), numRegionFaces * 3 / 4 );
        EXPECT_EQ( vertForms.size(), mesh.topology.vertSize() );
        for ( auto f : region )
        {
            EXPECT_TRUE( mesh.topology.hasFace( f ) );
            EXPECT_GT( mesh.triCenter( f ).z, -0.1f ); // the region still covers the upper half of the sphere
        }
        for ( auto ue : notFlippable )
            EXPECT_FALSE( mesh.topology.isLoneEdge( ue ) );
    }
}

TEST( MRMesh, MeshDecimateWithoutQualityChecks )
{
    for ( int parts : { 1, 8 } )
    {
        auto mesh = makeSphere( { .numMeshVertices = 2000 } );
        mesh.packOptimally( false );
        const auto numFaces = mesh.topology.numValidFaces();
        DecimateSettings settings;
        settings.maxError = FLT_MAX;
        settings.maxDeletedFaces = numFaces / 2;
        settings.maxTriangleAspectRatio = FLT_MAX;
        settings.preventNormalFlips = false;
        settings.subdivideParts = parts;
        const auto res = decimateMesh( mesh, settings );
        EXPECT_EQ( res.facesDeleted, numFaces / 2 );
        EXPECT_EQ( mesh.topology.numValidFaces(), numFaces - numFaces / 2 );
        EXPECT_TRUE( mesh.topology.checkValidity() );
    }
}

TEST( MRMesh, MeshDecimateMultipleEdgeResolve )
{
    //          2               /
    //         /|\              /
    //       // 5 \\            /
    //     / / / \ \ \          /
    //   /  //     \\  \        /
    //  3---6-------4---1       /
    //    \  \     /  /         /
    //       \ \ / /            /
    //          0               /

    // this test checks scenario where 46 cannot collapse due to creating multiple edge
    // but flipping 26->35 and 24->15 allows the collapse of 46
    VertCoords points( 7 );
    points[0_v] = Vector3f();
    points[1_v] = Vector3f( 3, 2, 0 );
    points[2_v] = Vector3f( 0, 6, 0 );
    points[3_v] = Vector3f( -3, 2, 0 );
    points[4_v] = Vector3f( 0, 2, 0 );
    points[5_v] = Vector3f( 0, 4, 0 );
    points[6_v] = Vector3f( 0, 2, 0 );
    // 46 is degenerate

    Triangulation t( 8 );
    t[0_f] = { 0_v,1_v,4_v };
    t[1_f] = { 0_v,4_v,6_v };
    t[2_f] = { 0_v,6_v,3_v };
    t[3_f] = { 4_v,1_v,2_v };
    t[4_f] = { 4_v,2_v,5_v };
    t[5_f] = { 4_v,5_v,6_v };
    t[6_f] = { 6_v,5_v,2_v };
    t[7_f] = { 6_v,2_v,3_v };
    Mesh mesh = Mesh::fromTriangles( points, t );

    DecimateSettings dsettings
    {
        .strategy = DecimateStrategy::ShortestEdgeFirst,
        .maxEdgeLen = 0.1f,
        .criticalTriAspectRatio = 1e4f,
        .tinyEdgeLength = 0.025f,
        .optimizeVertexPos = false, // this decreases probability of normal inversion near mesh degenerations
        .maxAngleChange = PI_F / 3
    };
    // only allow collapsing 46 edge
    dsettings.preCollapse = [&mesh] ( EdgeId e, const Vector3f )->bool
    {
        auto l = mesh.topology.left( e );
        auto r = mesh.topology.right( e );
        return ( l == 1_f && r == 5_f ) || ( l == 5_f && r == 1_f );
    };
    auto res = decimateMesh( mesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, 1 );

    // rotate triangulation in a way that 46 edge is further from beginning than 26 and 24 (to change decimation queue order)
    std::swap( t[1_f], t[7_f] );
    std::swap( t[5_f], t[6_f] );
    mesh = Mesh::fromTriangles( points, t ); // reversed
    // only allow to collapse 46 edge
    dsettings.preCollapse = [&mesh] ( EdgeId e, const Vector3f )->bool
    {
        auto l = mesh.topology.left( e );
        auto r = mesh.topology.right( e );
        return ( l == 6_f && r == 7_f ) || ( l == 7_f && r == 6_f );
    };
    res = decimateMesh( mesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, 1 );
}

static void testResolveDegen( const char * offMesh, float maxError, int vertsDeleted, int facesDeleted )
{
    std::istringstream s( offMesh );
    auto maybeMesh = MeshLoad::fromOff( s );
    EXPECT_TRUE( maybeMesh );

    DecimateSettings dsettings
    {
        .strategy = DecimateStrategy::ShortestEdgeFirst,
        .maxError = maxError,
        .criticalTriAspectRatio = 1000,
        .tinyEdgeLength = 0.1f * maxError,
        .stabilizer = 1e-6f,
        .optimizeVertexPos = false,
        .maxAngleChange = PI_F / 3
    };

    auto res = decimateMesh( *maybeMesh, dsettings );
    EXPECT_EQ( res.vertsDeleted, vertsDeleted );
    EXPECT_EQ( res.facesDeleted, facesDeleted );
}

TEST( MRMesh, MeshDecimateResolveDegen )
{
    // this case failed before special treatment of pockets in checkDeloneQuadrangle
    testResolveDegen
    (
        "OFF\n"
        "8 8 0\n"
        "\n"
        "0.25991842 0.11936638 0.40698087\n"
        "0.22987929 0.1279805 0.40149236\n"
        "0.25991842 0.11936638 0.4017116\n"
        "0.25991842 0.11936644 0.40171158\n"
        "0.25831458 0.113773495 0.395957\n"
        "0.25991842 0.11936644 0.39124623\n"
        "0.22980908 0.1280007 0.37024236\n"
        "0.25991842 0.11936638 0.37573087\n"
        "\n"
        "3 0 1 2\n"
        "3 3 4 0\n"
        "3 5 2 1\n"
        "3 6 7 5\n"
        "3 7 3 0\n"
        "3 7 4 3\n"
        "3 5 0 2\n"
        "3 0 5 7\n", 8e-6f, 1, 2
    );

    // this case failed before special treatment of pockets in checkDeloneQuadrangle
    testResolveDegen
    (
        "OFF\n"
        "9 8 0\n"
        "\n"
        "0.25991842 0.11936644 0.4636269\n"
        "0.23033604 0.12784958 0.46399236\n"
        "0.25991842 0.11936638 0.45374623\n"
        "0.25991842 0.11936638 0.46948087\n"
        "0.2422457 0.12443432 0.47700337\n"
        "0.25991842 0.11936638 0.4642116\n"
        "0.25991842 0.11936644 0.46421158\n"
        "0.25985792 0.11915526 0.46666718\n"
        "0.25979832 0.118947595 0.4617465\n"
        "\n"
        "3 0 1 2\n"
        "3 3 4 5\n"
        "3 0 5 4\n"
        "3 2 6 3\n"
        "3 7 3 6\n"
        "3 8 6 2\n"
        "3 0 3 5\n"
        "3 2 3 0\n", 8e-6f, 1, 2
    );

    // this case failed before special treatment of MultipleEdge error in MeshDecimate
    testResolveDegen
    (
        "OFF\n"
        "13 18 0\n"
        "\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.36842522 -0.0042799567 -0.0077246446\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.36411813 -0.019299522 -0.023349643\n"
        "0.36842522 -0.004279927 -0.0077246428\n"
        "0.3727323 0.010739669 -0.023349643\n"
        "0.36411813 -0.019299552 -0.023349643\n"
        "0.36842522 -0.004279927 -0.013974667\n"
        "0.3598111 -0.034319118 -0.0077246428\n"
        "0.36842522 -0.0042799567 -0.0077246428\n"
        "0.37273225 0.010739639 -0.023349643\n"
        "0.37273225 0.010739639 0.007900357\n"
        "0.36411813 -0.019299552 0.007900357\n"
        "\n"
        "3 0 1 2\n"
        "3 3 2 1\n"
        "3 4 1 5\n"
        "3 1 4 3\n"
        "3 0 5 1\n"
        "3 8 9 6\n"
        "3 10 9 11\n"
        "3 9 12 11\n"
        "3 4 0 7\n"
        "3 0 4 5\n"
        "3 7 0 9\n"
        "3 7 9 10\n"
        "3 9 2 6\n"
        "3 2 9 0\n"
        "3 2 7 6\n"
        "3 7 2 4\n"
        "3 4 2 3\n"
        "3 9 8 12\n", 3e-5f, 6, 12
    );
}

TEST( MRMesh, MeshDecimateOtherEnd )
{
    // prism of height 2 whose side has a tiny stair step: wall b-c, strip c-a 0.014 wide, wall a-e;
    // edge c-a is the cheapest to collapse, its optimal position and keeping c make too thin triangles, but keeping a removes the step
    const VertCoords points
    {
        { -0.2f, 1.1f, 2 }, { 0, 0.165f, 2 }, { 0, 0, 2 }, { 0.014f, 0, 2 }, { 0.6f, -10, 2 }, { 20, -10, 2 }, { 20, 40, 2 }, { -0.2f, 40, 2 }, // k b c a e f g h
        { -0.2f, 1.1f, 0 }, { 0, 0.165f, 0 }, { 0, 0, 0 }, { 0.6f, -10, 0 }, { 20, -10, 0 }, { 20, 40, 0 }, { -0.2f, 40, 0 }               // k0 b0 c0 e0 f0 g0 h0
    };
    const Triangulation t
    {
        { 2_v, 3_v, 1_v }, { 6_v, 1_v, 3_v }, { 4_v, 6_v, 3_v }, { 1_v, 6_v, 0_v }, { 4_v, 5_v, 6_v }, { 0_v, 6_v, 7_v },      // top
        { 9_v, 2_v, 1_v }, { 9_v, 10_v, 2_v }, { 2_v, 10_v, 3_v }, { 10_v, 4_v, 3_v }, { 10_v, 11_v, 4_v },                // b-c, strip c-a, a-e
        { 8_v, 9_v, 0_v }, { 0_v, 9_v, 1_v }, { 11_v, 12_v, 5_v }, { 11_v, 5_v, 4_v }, { 12_v, 13_v, 6_v }, { 12_v, 6_v, 5_v },  // k-b, e-f, f-g
        { 13_v, 14_v, 7_v }, { 13_v, 7_v, 6_v }, { 14_v, 8_v, 0_v }, { 14_v, 0_v, 7_v },                                    // g-h, h-k
        { 13_v, 9_v, 8_v }, { 13_v, 10_v, 9_v }, { 13_v, 11_v, 10_v }, { 13_v, 12_v, 11_v }, { 13_v, 8_v, 14_v }            // bottom
    };
    Mesh mesh = Mesh::fromTriangles( points, t );
    const auto res = decimateMesh( mesh, { .maxError = 0.03f, .stabilizer = 1e-5f } );
    EXPECT_EQ( res.vertsDeleted, 1 );
    EXPECT_FALSE( mesh.topology.hasVert( 2_v ) );
    EXPECT_EQ( mesh.points[3_v], points[3_v] );
}

TEST( MRMesh, DecimateFixedBoundary )
{
    // half of a torus, whose two boundary circles shall stay unchanged
    const Mesh mesh0 = []
    {
        Mesh m = makeTorus( 1.0f, 0.3f, 64, 16 );
        FaceBitSet half( m.topology.faceSize() );
        for ( auto f : m.topology.getValidFaces() )
            if ( m.triCenter( f ).x < 0 )
                half.set( f );
        m.topology.deleteFaces( half );
        return m;
    }();
    auto bdVerts = getBoundaryVerts( mesh0.topology );
    const auto len = mesh0.averageEdgeLength();
    auto expectFixedBd = [&]( const Mesh & mesh )
    {
        int changed = 0;
        for ( auto v : bdVerts )
            if ( !mesh.topology.hasVert( v ) || !mesh.topology.isBdVertex( v ) || mesh.points[v] != mesh0.points[v] )
                ++changed;
        EXPECT_EQ( changed, 0 );
    };

    Mesh mesh = mesh0;
    remesh( mesh, { .targetEdgeLen = 3 * len, .frozenBoundary = true } );
    expectFixedBd( mesh );

    for ( auto strategy : { DecimateStrategy::MinimizeError, DecimateStrategy::ShortestEdgeFirst } )
        for ( int parts : { 1, 8 } )
            for ( auto givenBd : { (VertBitSet*)nullptr, &bdVerts } )
            {
                SCOPED_TRACE( testing::Message() << "strategy " << int( strategy ) << ", parts " << parts << ", bdVerts " << ( givenBd != nullptr ) );
                mesh = mesh0;
                const auto res = decimateMesh( mesh, { .strategy = strategy, .maxError = strategy == DecimateStrategy::MinimizeError ? 0.01f : 3 * len,
                    .touchBdVerts = false, .bdVerts = givenBd, .subdivideParts = parts } );
                EXPECT_GT( res.vertsDeleted, 0 );
                expectFixedBd( mesh );
            }
}

TEST( MRMesh, DecimateObjectMeshDataShortAttributes )
{
    const Mesh original = makeSphere( { .numMeshVertices = 400 } );
    ObjectMeshData data;
    data.mesh = std::make_shared<Mesh>( original );
    // gaps in face ids, so that packing moves faces
    FaceBitSet del( original.topology.faceSize() );
    for ( FaceId f( 0 ); f < 20; ++f )
        del.set( f );
    data.mesh->deleteFaces( del );
    const auto & t = data.mesh->topology;

    // no color and texture for the last face, and no uv for the last vertex
    data.faceColors.resize( size_t( t.lastValidFace() ) );
    for ( FaceId f( 0 ); f < data.faceColors.size(); ++f )
        data.faceColors[f] = Color( int( f ) % 256, int( f ) / 256, 0 );
    data.uvCoordinates.resize( size_t( t.lastValidVert() ) );
    for ( VertId v( 0 ); v < data.uvCoordinates.size(); ++v )
        data.uvCoordinates[v] = { float( v ), 0.0f };
    data.texturePerFace.resize( size_t( t.lastValidFace() ) );
    for ( FaceId f( 0 ); f < data.texturePerFace.size(); ++f )
        data.texturePerFace[f] = TextureId( int( f ) % 3 + 1 );
    const auto before = data.clone();
    std::map<std::tuple<float, float, float>, VertId> vertAt;
    for ( auto v : t.getValidVerts() )
    {
        const auto p = data.mesh->points[v];
        vertAt[{ p.x, p.y, p.z }] = v;
    }

    decimateObjectMeshData( data, { .maxDeletedVertices = 0, .packMesh = true } );

    // only the packing has happened, and old elements kept their attributes
    ASSERT_EQ( t.numValidFaces(), before.mesh->topology.numValidFaces() );
    EXPECT_EQ( data.faceColors.size(), t.faceSize() );
    EXPECT_EQ( data.texturePerFace.size(), t.faceSize() );
    EXPECT_EQ( data.uvCoordinates.size(), t.vertSize() );
    for ( auto f : t.getValidFaces() )
    {
        const auto pf = findProjection( data.mesh->triCenter( f ), *before.mesh ).proj.face;
        if ( pf < before.faceColors.size() )
        {
            EXPECT_EQ( data.faceColors[f], before.faceColors[pf] );
            EXPECT_EQ( data.texturePerFace[f], before.texturePerFace[pf] );
        }
        else
        {
            // the face without texture gets the first one, not an invalid id
            EXPECT_EQ( data.texturePerFace[f], TextureId( 0 ) );
        }
    }
    for ( auto v : t.getValidVerts() )
    {
        const auto p = data.mesh->points[v];
        const auto pv = vertAt.at( { p.x, p.y, p.z } );
        if ( pv < before.uvCoordinates.size() )
        {
            EXPECT_EQ( data.uvCoordinates[v], before.uvCoordinates[pv] );
        }
    }
}

TEST( MRMesh, RemeshTargetTriCountBeyondInt )
{
    // area / targetTriArea ~ 2e13 does not fit in int: nothing to decimate, and no overflow
    Mesh sphere = makeUVSphere( 1000.f, 8, 8 );
    const auto numFaces = sphere.topology.numValidFaces();
    RemeshSettings rs;
    rs.maxEdgeSplits = 100;
    EXPECT_TRUE( remesh( sphere, rs ) );
    EXPECT_TRUE( sphere.topology.checkValidity() );
    EXPECT_GT( sphere.topology.numValidFaces(), numFaces );
}

TEST( MRMesh, RemeshKeepsSharpEdges )
{
    // edge flips across the cube's edges used to be limited only by the absolute deviation 1, losing 1/3 of the volume
    const Mesh cube = makeCube();
    Mesh mesh = cube;
    RemeshSettings rs;
    rs.targetEdgeLen = 0.05f;
    rs.maxAngleChangeAfterFlip = PI_F;
    EXPECT_TRUE( remesh( mesh, rs ) );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_GT( mesh.topology.numValidFaces(), 1000 );
    EXPECT_NEAR( mesh.volume(), 1.0, 0.01 );
    float maxDist = 0;
    for ( auto v : mesh.topology.getValidVerts() )
        maxDist = std::max( maxDist, std::sqrt( findProjection( mesh.points[v], cube ).distSq ) );
    EXPECT_LT( maxDist, rs.targetEdgeLen );
}

TEST( MRMesh, MeshDecimateNoFold )
{
    // box [0,1]x[0,1]x[0,2] with tiny top triangle (c,a,b) at the corner (0,0,2);
    // moving b onto a deletes it, side triangle (b0,c,b) turns by almost 90 degrees and folds onto its new neighbour (c0,a,c)
    const float d = 0.01f, h = 0.2f;
    const VertCoords points
    {
        { 0, 0, 2 }, { d, 0.001f, 2 }, { 0, h, 2 }, { 1, 0, 2 }, { 1, 1, 2 }, { 0, 1, 2 }, // c, a, b, top corners
        { 0, 0, 0 }, { d, 0, 0 }, { 0, h, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }        // c0, a0, b0, bottom corners
    };
    const Triangulation t
    {
        { 0_v, 1_v, 2_v }, { 1_v, 3_v, 4_v }, { 1_v, 4_v, 2_v }, { 2_v, 4_v, 5_v },      // top
        { 6_v, 7_v, 1_v }, { 6_v, 1_v, 0_v }, { 7_v, 9_v, 3_v }, { 7_v, 3_v, 1_v },      // y = 0
        { 8_v, 0_v, 2_v }, { 6_v, 0_v, 8_v }, { 8_v, 2_v, 5_v }, { 8_v, 5_v, 11_v },     // x = 0
        { 9_v, 10_v, 4_v }, { 9_v, 4_v, 3_v }, { 10_v, 11_v, 5_v }, { 10_v, 5_v, 4_v },  // x = 1, y = 1
        { 10_v, 9_v, 7_v }, { 10_v, 7_v, 6_v }, { 10_v, 6_v, 8_v }, { 10_v, 8_v, 11_v }  // bottom
    };
    Mesh mesh = Mesh::fromTriangles( points, t );
    ASSERT_TRUE( mesh.topology.isClosed() );

    UndirectedEdgeBitSet edgesToCollapse( mesh.topology.undirectedEdgeSize() );
    edgesToCollapse.set( mesh.topology.findEdge( 1_v, 2_v ).undirected() );
    DecimateSettings settings
    {
        .maxError = 0.05f, // allows moving b onto a, but not a onto b
        .maxTriangleAspectRatio = FLT_MAX,
        .optimizeVertexPos = false,
        .edgesToCollapse = &edgesToCollapse
    };
    decimateMesh( mesh, settings );
    for ( auto ue : undirectedEdges( mesh.topology ) )
        EXPECT_GT( dot( mesh.normal( mesh.topology.left( ue ) ), mesh.normal( mesh.topology.right( ue ) ) ), -0.5f );
}

} //namespace MR
