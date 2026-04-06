#include <MRMesh/MRMeshFillHole.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMeshFixer.h>
#include <MRMesh/MRGTest.h>

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

} //namespace MR
