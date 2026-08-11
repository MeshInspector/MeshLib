#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRRingIterator.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, stitchHoles )
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );

    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // VertId{0}
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // VertId{1}
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // VertId{2}
    mesh.points.emplace_back( 0.f, 0.f, 1.f ); // VertId{3}
    mesh.points.emplace_back( 1.f, 0.f, 1.f ); // VertId{4}
    mesh.points.emplace_back( 0.f, 1.f, 1.f ); // VertId{5}
    EXPECT_EQ( mesh.points.size(), 6 );

    auto bdEdges = mesh.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 2 );
    EXPECT_FALSE( mesh.topology.left( bdEdges[0] ).valid() );
    EXPECT_FALSE( mesh.topology.left( bdEdges[1] ).valid() );

    FaceBitSet newFaces;
    StitchHolesParams params;
    auto fsz0 = mesh.topology.faceSize();
    params.outNewFaces = &newFaces;
    stitchHoles( mesh, bdEdges[0], bdEdges[1], params );
    auto numNewFaces = mesh.topology.faceSize() - fsz0;

    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 8 );
    EXPECT_EQ( mesh.points.size(), 6 );
    EXPECT_EQ( numNewFaces, 6 );
    EXPECT_EQ( newFaces.count(), 6 );
    EXPECT_EQ( newFaces.size(), 8 );

    bdEdges = mesh.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 0 );
}

TEST( MRMesh, makeBridge )
{
    MeshTopology topology;
    auto a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    auto b = topology.makeEdge();
    topology.setOrg( b, topology.addVertId() );
    topology.setOrg( b.sym(), topology.addVertId() );
    EXPECT_EQ( topology.numValidFaces(), 0 );
    FaceBitSet fbs;
    auto bridgeRes = makeBridge( topology, a, b, &fbs );
    EXPECT_TRUE( bridgeRes );
    EXPECT_EQ( bridgeRes.newFaces, 2 );
    EXPECT_TRUE( bridgeRes.na );
    EXPECT_EQ( topology.org( a ), topology.org( bridgeRes.na ) );
    EXPECT_TRUE( topology.left( a ) );
    EXPECT_FALSE( topology.left( bridgeRes.na ) );
    EXPECT_TRUE( bridgeRes.nb );
    EXPECT_EQ( topology.org( b ), topology.org( bridgeRes.nb ) );
    EXPECT_TRUE( topology.left( b ) );
    EXPECT_FALSE( topology.left( bridgeRes.nb ) );
    EXPECT_EQ( fbs.count(), 2 );
    EXPECT_EQ( topology.numValidVerts(), 4 );
    EXPECT_EQ( topology.numValidFaces(), 2 );
    EXPECT_EQ( topology.edgeSize(), 5 * 2 );

    topology = MeshTopology();
    a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    b = topology.makeEdge();
    topology.splice( a.sym(), b );
    topology.setOrg( b.sym(), topology.addVertId() );
    EXPECT_EQ( topology.numValidFaces(), 0 );
    fbs.reset();
    bridgeRes = makeBridge( topology, a, b, &fbs );
    EXPECT_TRUE( bridgeRes );
    EXPECT_EQ( bridgeRes.newFaces, 1 );
    EXPECT_TRUE( bridgeRes.na );
    EXPECT_EQ( topology.org( a ), topology.org( bridgeRes.na ) );
    EXPECT_TRUE( topology.left( a ) );
    EXPECT_FALSE( topology.left( bridgeRes.na ) );
    EXPECT_FALSE( bridgeRes.nb );
    EXPECT_TRUE( topology.left( b ) );
    EXPECT_EQ( fbs.count(), 1 );
    EXPECT_EQ( topology.numValidVerts(), 3 );
    EXPECT_EQ( topology.numValidFaces(), 1 );
    EXPECT_EQ( topology.edgeSize(), 3 * 2 );
}

TEST( MRMesh, makeBridgeEdge )
{
    MeshTopology topology;
    auto a = topology.makeEdge();
    topology.setOrg( a, topology.addVertId() );
    topology.setOrg( a.sym(), topology.addVertId() );
    auto b = topology.makeEdge();
    topology.setOrg( b, topology.addVertId() );
    topology.setOrg( b.sym(), topology.addVertId() );
    auto x = makeBridgeEdge( topology, a, b );
    EXPECT_TRUE( topology.fromSameOriginRing( a, x ) );
    EXPECT_TRUE( topology.fromSameOriginRing( b, x.sym() ) );
    EXPECT_EQ( topology.edgeSize(), 3 * 2 );

    x = makeBridgeEdge( topology, a, b );
    EXPECT_FALSE( x.valid() );
}

TEST( MRMesh, makeInterHoleBridgeEdges )
{
    // two separate triangles, one on top of the other with opposite orientations, with a hole around each
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 3_v, 5_v, 4_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f ); // VertId{0}
    mesh.points.emplace_back( 1.f, 0.f, 0.f ); // VertId{1}
    mesh.points.emplace_back( 0.f, 1.f, 0.f ); // VertId{2}
    mesh.points.emplace_back( 0.f, 0.f, 1.f ); // VertId{3}
    mesh.points.emplace_back( 1.f, 0.f, 1.f ); // VertId{4}
    mesh.points.emplace_back( 0.f, 1.f, 1.f ); // VertId{5}

    auto bdEdges = mesh.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 2 );

    // no bridges if less than two holes are given
    EXPECT_TRUE( makeInterHoleBridgeEdges( mesh, {} ).empty() );
    EXPECT_TRUE( makeInterHoleBridgeEdges( mesh, { bdEdges[0] } ).empty() );

    // a bridge appears between each pair of mutually closest vertices: (0,3), (1,4), (2,5)
    auto bridges = makeInterHoleBridgeEdges( mesh, bdEdges );
    EXPECT_EQ( bridges.size(), 3 );
    for ( EdgeId b : bridges )
    {
        EXPECT_FALSE( mesh.topology.left( b ).valid() );
        EXPECT_FALSE( mesh.topology.right( b ).valid() );
        const auto d = mesh.destPnt( b ) - mesh.orgPnt( b );
        EXPECT_EQ( d.x, 0.f );
        EXPECT_EQ( d.y, 0.f );
        EXPECT_EQ( d.lengthSq(), 1.f );
    }
    EXPECT_EQ( mesh.topology.numValidVerts(), 6 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), 3 );

    // vertices 0 and 3 are mutually closest, but no bridge is created between them,
    // because it would go deep inside the triangle 0-1-2 incident to vertex 0
    Triangulation t2{
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v }
    };
    Mesh mesh2;
    mesh2.topology = MeshBuilder::fromTriangles( t2 );
    mesh2.points.emplace_back(   0.f, 0.f, 0.f ); // VertId{0}
    mesh2.points.emplace_back(  10.f, 0.f, 1.f ); // VertId{1}
    mesh2.points.emplace_back( -10.f, 0.f, 1.f ); // VertId{2}
    mesh2.points.emplace_back(   0.f, 0.f, 2.f ); // VertId{3}
    mesh2.points.emplace_back(   1.f, 3.f, 3.f ); // VertId{4}
    mesh2.points.emplace_back(  -1.f, 3.f, 3.f ); // VertId{5}

    bdEdges = mesh2.topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 2 );
    EXPECT_TRUE( makeInterHoleBridgeEdges( mesh2, bdEdges ).empty() );
}

TEST( MRMesh, HoleFillPlan3 )
{
    Mesh mesh;
    const auto e = mesh.addSeparateEdgeLoop
    ( {
        {  0, -1, 0 },
        {  2,  0, 0 },
        {  0,  1, 0 }
    } );

    auto p0 = getPlanarHoleFillPlan( mesh, e );
    EXPECT_EQ( p0.items.size(), 0 );
    EXPECT_EQ( p0.numTris, 1 );

    auto p1 = getPlanarHoleFillPlan( mesh, e.sym() );
    EXPECT_EQ( p1.items.size(), 0 );
    EXPECT_EQ( p1.numTris, 1 );

    EXPECT_TRUE( isFillingMultipleEdgeFree( mesh.topology, p0 ) );
    executeHoleFillPlan( mesh, e, p0 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 1 );
    EXPECT_FALSE( mesh.topology.isClosed() );

    EXPECT_TRUE( isFillingMultipleEdgeFree( mesh.topology, p1 ) );
    executeHoleFillPlan( mesh, e.sym(), p1 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_TRUE( mesh.topology.isClosed() );
}

TEST( MRMesh, HoleFillPlan4 )
{
    Mesh mesh;
    const auto e = mesh.addSeparateEdgeLoop
    ( {
        {  0, -1, 0 },
        {  2,  0, 0 },
        {  0,  1, 0 },
        { -2,  0, 0 }
    } );

    auto p0 = getPlanarHoleFillPlan( mesh, e );
    EXPECT_EQ( p0.items.size(), 1 );
    EXPECT_EQ( p0.numTris, 2 );

    auto p1 = getPlanarHoleFillPlan( mesh, e.sym() );
    EXPECT_EQ( p1.items.size(), 1 );
    EXPECT_EQ( p1.numTris, 2 );

    EXPECT_TRUE( isFillingMultipleEdgeFree( mesh.topology, p0 ) );
    executeHoleFillPlan( mesh, e, p0 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_FALSE( mesh.topology.isClosed() );
    EXPECT_FALSE( hasMultipleEdges( mesh.topology ) );

    auto mesh1 = mesh;

    // independently produced plans can result in multiple edges after execution:
    EXPECT_FALSE( isFillingMultipleEdgeFree( mesh.topology, p1 ) );
    executeHoleFillPlan( mesh, e.sym(), p1 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_TRUE( hasMultipleEdges( mesh.topology ) );

    // if the plan to fill the second hole is prepared after the first hole is filled, no multiple edges appear
    auto p11 = getPlanarHoleFillPlan( mesh1, e.sym() );
    EXPECT_EQ( p11.items.size(), 1 );
    EXPECT_EQ( p11.numTris, 2 );
    EXPECT_TRUE( isFillingMultipleEdgeFree( mesh1.topology, p11 ) );
    executeHoleFillPlan( mesh1, e.sym(), p11 );
    EXPECT_EQ( mesh1.topology.numValidFaces(), 4 );
    EXPECT_TRUE( mesh1.topology.isClosed() );
    EXPECT_FALSE( hasMultipleEdges( mesh1.topology ) );
}

// hexagonal hole, and the edges of the hole to the left of the returned one
static Mesh makeHexagonHole( EdgeId & e, std::vector<EdgeId> & holeEdges )
{
    Mesh mesh;
    e = mesh.addSeparateEdgeLoop
    ( {
        {  2,  0, 0 },
        {  1,  2, 0 },
        { -1,  2, 0 },
        { -2,  0, 0 },
        { -1, -2, 0 },
        {  1, -2, 0 }
    } );
    holeEdges.clear();
    for ( auto ei : leftRing( mesh.topology, e ) )
        holeEdges.push_back( ei );
    return mesh;
}

TEST( MRMesh, HoleFillPlanEdgesOnly )
{
    EdgeId e;
    std::vector<EdgeId> he;
    auto mesh = makeHexagonHole( e, he );
    ASSERT_EQ( he.size(), 6 );
    std::vector<VertId> v;
    for ( auto ei : he )
        v.push_back( mesh.topology.org( ei ) );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), 2 );

    // numTris stays zero: the plan only splits the hole in parts and creates no face.
    // the second chord starts where the first one ends, which is expressible only as the sym
    // of the first item: -( 2 * item + sym + 1 ) == -2
    HoleFillPlan plan;
    plan.items.push_back( { (int)he[2], (int)he[0] } ); // chord org( he2 ) -> org( he0 )
    plan.items.push_back( { -2, (int)he[4] } );         // chord org( he0 ) -> org( he4 )

    executeHoleFillPlan( mesh, e, plan );
    EXPECT_EQ( mesh.topology.numValidFaces(), 0 );
    EXPECT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), 4 ); // 3 parts and the other side
    EXPECT_TRUE( mesh.topology.findEdge( v[2], v[0] ).valid() );
    EXPECT_TRUE( mesh.topology.findEdge( v[0], v[4] ).valid() );
    // without the sym bit the second chord would have started at the other end of the first one
    EXPECT_FALSE( mesh.topology.findEdge( v[2], v[4] ).valid() );
}

TEST( MRMesh, HoleFillPlanSymAnchorMultipleEdge )
{
    EdgeId e;
    std::vector<EdgeId> he;
    auto mesh = makeHexagonHole( e, he );
    ASSERT_EQ( he.size(), 6 );
    const auto v0 = mesh.topology.org( he[0] );
    const auto v4 = mesh.topology.org( he[4] );

    HoleFillPlan pre;
    pre.items.push_back( { (int)he[0], (int)he[4] } );
    executeHoleFillPlan( mesh, e, pre );
    ASSERT_TRUE( mesh.topology.findEdge( v0, v4 ).valid() );

    // the second item would add org( he0 ) -> org( he4 ) once more, which already exists;
    // resolving its sym code by the first item's edgeCode1 would look at the other end and miss it
    HoleFillPlan plan;
    plan.items.push_back( { (int)he[2], (int)he[0] } );
    plan.items.push_back( { -2, (int)he[4] } );
    EXPECT_FALSE( isFillingMultipleEdgeFree( mesh.topology, plan ) );
}

} //namespace MR
