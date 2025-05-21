#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRBitSetParallelFor.h>

namespace MR
{

TEST(MRMesh, Pack)
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    Mesh mesh;
    EXPECT_TRUE( mesh.topology.checkValidity() );

    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 1.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );

    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 4 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 2 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(9) ); // 5*2 = 10 half-edges in total

    Mesh dbl = mesh;
    dbl.addMesh( mesh );
    EXPECT_TRUE( dbl.topology.checkValidity() );
    EXPECT_EQ( dbl.points.size(), 8 );
    EXPECT_EQ( dbl.topology.numValidVerts(), 8 );
    EXPECT_EQ( dbl.topology.numValidFaces(), 4 );
    EXPECT_EQ( dbl.topology.lastNotLoneEdge(), EdgeId(19) ); // 10*2 = 20 half-edges in total

    mesh.topology.deleteFace( 1_f );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 4 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 2 );

    mesh.pack();
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.points.size(), 3 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 3 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 3 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 1 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 1 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId(5) ); // 3*2 = 6 half-edges in total

    mesh.topology.deleteFace( FaceId( 0 ) );
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.points.size(), 3 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 3 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 1 );

    mesh.pack();
    EXPECT_TRUE( mesh.topology.checkValidity() );
    EXPECT_EQ( mesh.points.size(), 0 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 0 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 0 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), EdgeId() );
}

TEST(MRMesh, Pack2)
{
    auto mesh = makeCube();
    mesh.topology.deleteFace( 2_f );
    mesh.topology.deleteFace( 5_f );
    mesh.invalidateCaches();
    EXPECT_EQ( mesh.topology.faceSize(), 12 );
    EXPECT_EQ( mesh.topology.vertSize(), 8 );
    EXPECT_EQ( mesh.topology.undirectedEdgeSize(), 18 );

    auto fs = mesh.topology.getValidFaces();

    FaceMap packedToOrgFace;
    mesh.pack( Tgt2SrcMaps( &packedToOrgFace, nullptr, nullptr ) );
    EXPECT_EQ( mesh.topology.faceSize(), 10 );
    EXPECT_EQ( mesh.topology.vertSize(), 8 );
    EXPECT_EQ( mesh.topology.undirectedEdgeSize(), 17 );

    FaceBitSet fsAfterPack( mesh.topology.numValidFaces(), false );
    BitSetParallelForAll( fsAfterPack, [&]( FaceId f )
    {
        if ( fs.test( packedToOrgFace[f] ) )
            fsAfterPack.set( f );
    });
    EXPECT_EQ( fsAfterPack, mesh.topology.getValidFaces() );
}

TEST(MRMesh, AddPartByMask) 
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

    EXPECT_EQ( mesh.points.size(), 4 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 4 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 4 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 2 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 2 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), 9_e ); // 5*2 = 10 half-edges in total

    Mesh mesh2 = mesh;
    FaceBitSet faces( 2 );
    faces.set( 1_f );

    auto meshIntoMesh2 = FaceMapOrHashMap::createHashMap();
    FaceMapOrHashMap mesh2IntoMesh;
    PartMapping mapping;
    mapping.src2tgtFaces = &meshIntoMesh2;
    mapping.tgt2srcFaces = &mesh2IntoMesh;

    mesh.addMeshPart( { mesh2, &faces }, mapping );
    EXPECT_TRUE( meshIntoMesh2.getHashMap() != nullptr );
    for ( auto [f, f2] : *meshIntoMesh2.getHashMap() )
        EXPECT_EQ( getAt( mesh2IntoMesh, f2 ), f );

    faces.set( 0_f ); // set an id without mapping
    auto added = faces.getMapping( *meshIntoMesh2.getHashMap() );

    EXPECT_EQ( mesh.points.size(), 7 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 7 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 7 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 3 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 3 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), 15_e ); // 8*2 = 16 half-edges in total

    faces.set( 0_f );
    faces.reset( 1_f );
    mesh.addMeshPart( { mesh2, &faces } );

    EXPECT_EQ( mesh.points.size(), 10 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 10 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 10 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), 21_e ); // 11*2 = 22 half-edges in total
}

TEST(MRMesh, AddPartByMaskAndStitch) 
{
    Triangulation t{ { 0_v, 1_v, 2_v } };
    auto topology0 = MeshBuilder::fromTriangles( t );
    auto topology1 = topology0;

    // stitch along open contour
    std::vector<EdgePath> c0 = { { topology0.findEdge( 1_v, 0_v ) } };
    std::vector<EdgePath> c1 = { { topology1.findEdge( 0_v, 1_v ) } };
    auto topologyRes = topology0;
    topologyRes.addPartByMask( topology1, topology1.getValidFaces(), false, c0, c1 );
    EXPECT_TRUE( topologyRes.checkValidity() );
    EXPECT_EQ( topologyRes.numValidVerts(), 4 );
    EXPECT_EQ( topologyRes.numValidFaces(), 2 );
    EXPECT_EQ( topologyRes.lastNotLoneEdge(), 9_e ); // 5*2 = 10 half-edges in total

    // stitch along closed contour
    c0 = { { topology0.findEdge( 1_v, 0_v ) }, { topology0.findEdge( 0_v, 2_v ) }, { topology0.findEdge( 2_v, 1_v ) } };
    c1 = { { topology1.findEdge( 0_v, 1_v ) }, { topology1.findEdge( 1_v, 2_v ) }, { topology1.findEdge( 2_v, 0_v ) } };
    topologyRes = topology0;
    topologyRes.addPartByMask( topology1, topology1.getValidFaces(), false, c0, c1 );
    EXPECT_TRUE( topologyRes.checkValidity() );
    EXPECT_EQ( topologyRes.numValidVerts(), 3 );
    EXPECT_EQ( topologyRes.numValidFaces(), 2 );
    EXPECT_EQ( topologyRes.lastNotLoneEdge(), 5_e ); // 3*2 = 6 half-edges in total
}

TEST(MRMesh, AddMesh)
{
    auto cube = makeCube();
    cube.topology.deleteFace( 0_f );
    cube.topology.deleteFace( 1_f );
    cube.invalidateCaches();

    Mesh tgt;
    tgt.addMesh( cube );
    const auto nf = tgt.topology.faceSize();
    const auto nv = tgt.topology.vertSize();
    const auto ne = tgt.topology.undirectedEdgeSize();
    EXPECT_EQ( nf, 10 );
    EXPECT_EQ( nv, 8 );
    EXPECT_EQ( ne, 17 );

    auto faceMap = FaceMapOrHashMap::createMap();
    auto vertMap = VertMapOrHashMap::createMap();
    auto edgeMap = WholeEdgeMapOrHashMap::createMap();

    auto faceHashMap = FaceMapOrHashMap::createHashMap();
    auto vertHashMap = VertMapOrHashMap::createHashMap();
    auto edgeHashMap = WholeEdgeMapOrHashMap::createHashMap();

    tgt.addMesh( cube,
    {
        .src2tgtFaces = &faceMap,
        .src2tgtVerts = &vertMap,
        .src2tgtEdges = &edgeMap,
        .tgt2srcFaces = &faceHashMap,
        .tgt2srcVerts = &vertHashMap,
        .tgt2srcEdges = &edgeHashMap
    } );
    EXPECT_EQ( 2 * nf, tgt.topology.faceSize() );
    EXPECT_EQ( 2 * nv, tgt.topology.vertSize() );
    EXPECT_EQ( 2 * ne, tgt.topology.undirectedEdgeSize() );
    EXPECT_EQ( faceMap.getMap()->size(), 12 );
    EXPECT_EQ( vertMap.getMap()->size(), 8 );
    EXPECT_EQ( edgeMap.getMap()->size(), 18 );
    EXPECT_EQ( faceHashMap.getHashMap()->size(), nf );
    EXPECT_EQ( vertHashMap.getHashMap()->size(), nv );
    EXPECT_EQ( edgeHashMap.getHashMap()->size(), ne );

    faceMap.clear();
    vertMap.clear();
    edgeMap.clear();

    faceHashMap.clear();
    vertHashMap.clear();
    edgeHashMap.clear();

    tgt.addMesh( cube,
    {
        .src2tgtFaces = &faceHashMap,
        .src2tgtVerts = &vertHashMap,
        .src2tgtEdges = &edgeHashMap,
        .tgt2srcFaces = &faceMap,
        .tgt2srcVerts = &vertMap,
        .tgt2srcEdges = &edgeMap
    } );
    EXPECT_EQ( 3 * nf, tgt.topology.faceSize() );
    EXPECT_EQ( 3 * nv, tgt.topology.vertSize() );
    EXPECT_EQ( 3 * ne, tgt.topology.undirectedEdgeSize() );
    EXPECT_EQ( faceMap.getMap()->size(), 3 * nf );
    EXPECT_EQ( vertMap.getMap()->size(), 3 * nv );
    EXPECT_EQ( edgeMap.getMap()->size(), 3 * ne );
    EXPECT_EQ( faceHashMap.getHashMap()->size(), nf );
    EXPECT_EQ( vertHashMap.getHashMap()->size(), nv );
    EXPECT_EQ( edgeHashMap.getHashMap()->size(), ne );
}

TEST(MRMesh, AddMeshPart)
{
    const auto cube = makeCube();
    FaceBitSet fs( cube.topology.faceSize(), true );
    fs.reset( 0_f );
    fs.reset( 1_f );
    const MeshPart cubePart( cube, &fs );

    Mesh tgt;
    tgt.addMeshPart( cubePart );
    const auto nf = tgt.topology.faceSize();
    const auto nv = tgt.topology.vertSize();
    const auto ne = tgt.topology.undirectedEdgeSize();
    EXPECT_EQ( nf, 10 );
    EXPECT_EQ( nv, 8 );
    EXPECT_EQ( ne, 17 );

    auto faceMap = FaceMapOrHashMap::createMap();
    auto vertMap = VertMapOrHashMap::createMap();
    auto edgeMap = WholeEdgeMapOrHashMap::createMap();

    auto faceHashMap = FaceMapOrHashMap::createHashMap();
    auto vertHashMap = VertMapOrHashMap::createHashMap();
    auto edgeHashMap = WholeEdgeMapOrHashMap::createHashMap();

    tgt.addMeshPart( cubePart,
    {
        .src2tgtFaces = &faceMap,
        .src2tgtVerts = &vertMap,
        .src2tgtEdges = &edgeMap,
        .tgt2srcFaces = &faceHashMap,
        .tgt2srcVerts = &vertHashMap,
        .tgt2srcEdges = &edgeHashMap
    } );
    EXPECT_EQ( 2 * nf, tgt.topology.faceSize() );
    EXPECT_EQ( 2 * nv, tgt.topology.vertSize() );
    EXPECT_EQ( 2 * ne, tgt.topology.undirectedEdgeSize() );
    EXPECT_EQ( faceMap.getMap()->size(), 12 );
    EXPECT_EQ( vertMap.getMap()->size(), 8 );
    EXPECT_EQ( edgeMap.getMap()->size(), 18 );
    EXPECT_EQ( faceHashMap.getHashMap()->size(), nf );
    EXPECT_EQ( vertHashMap.getHashMap()->size(), nv );
    EXPECT_EQ( edgeHashMap.getHashMap()->size(), ne );

    faceMap.clear();
    vertMap.clear();
    edgeMap.clear();

    faceHashMap.clear();
    vertHashMap.clear();
    edgeHashMap.clear();

    tgt.addMeshPart( cubePart,
    {
        .src2tgtFaces = &faceHashMap,
        .src2tgtVerts = &vertHashMap,
        .src2tgtEdges = &edgeHashMap,
        .tgt2srcFaces = &faceMap,
        .tgt2srcVerts = &vertMap,
        .tgt2srcEdges = &edgeMap
    } );
    EXPECT_EQ( 3 * nf, tgt.topology.faceSize() );
    EXPECT_EQ( 3 * nv, tgt.topology.vertSize() );
    EXPECT_EQ( 3 * ne, tgt.topology.undirectedEdgeSize() );
    EXPECT_EQ( faceMap.getMap()->size(), 3 * nf );
    EXPECT_EQ( vertMap.getMap()->size(), 3 * nv );
    EXPECT_EQ( edgeMap.getMap()->size(), 3 * ne );
    EXPECT_EQ( faceHashMap.getHashMap()->size(), nf );
    EXPECT_EQ( vertHashMap.getHashMap()->size(), nv );
    EXPECT_EQ( edgeHashMap.getHashMap()->size(), ne );
}

} //namespace MR
