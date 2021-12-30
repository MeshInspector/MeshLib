#include "MRMesh.h"
#include "MRMeshBuilder.h"
#include "MRMeshDelete.h"
#include "MRBitSet.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, Pack) 
{
    std::vector<VertId> v{ 
        VertId{0}, VertId{1}, VertId{2}, 
        VertId{0}, VertId{2}, VertId{3}
    };
    Mesh mesh;
    EXPECT_TRUE( mesh.topology.checkValidity() );

    mesh.topology = MeshBuilder::fromVertexTriples( v );
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
    dbl.addPart( mesh );
    EXPECT_TRUE( dbl.topology.checkValidity() );
    EXPECT_EQ( dbl.points.size(), 8 );
    EXPECT_EQ( dbl.topology.numValidVerts(), 8 );
    EXPECT_EQ( dbl.topology.numValidFaces(), 4 );
    EXPECT_EQ( dbl.topology.lastNotLoneEdge(), EdgeId(19) ); // 10*2 = 20 half-edges in total

    deleteFace( mesh.topology, FaceId( 1 ) );
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

    deleteFace( mesh.topology, FaceId( 0 ) );
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

TEST(MRMesh, AddPartByMask) 
{
    std::vector<VertId> v{ 
        0_v, 1_v, 2_v, 
        0_v, 2_v, 3_v
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromVertexTriples( v );
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

    FaceHashMap meshIntoMesh2;
    FaceMap mesh2IntoMesh;
    PartMapping mapping;
    mapping.src2tgtFaces = &meshIntoMesh2;
    mapping.tgt2srcFaces = &mesh2IntoMesh;

    mesh.addPartByMask( mesh2, faces, mapping );
    for ( auto [f, f2] : meshIntoMesh2 )
        EXPECT_EQ( mesh2IntoMesh[f2], f );

    faces.set( 0_f ); // set an id without mapping
    auto added = faces.getMapping( meshIntoMesh2 );

    EXPECT_EQ( mesh.points.size(), 7 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 7 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 7 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 3 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 3 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), 15_e ); // 8*2 = 16 half-edges in total

    faces.set( 0_f );
    faces.reset( 1_f );
    mesh.addPartByMask( mesh2, faces );

    EXPECT_EQ( mesh.points.size(), 10 );
    EXPECT_EQ( mesh.topology.edgePerVertex().size(), 10 );
    EXPECT_EQ( mesh.topology.numValidVerts(), 10 );
    EXPECT_EQ( mesh.topology.edgePerFace().size(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 4 );
    EXPECT_EQ( mesh.topology.lastNotLoneEdge(), 21_e ); // 11*2 = 22 half-edges in total
}

TEST(MRMesh, AddPartByMaskAndStitch) 
{
    std::vector<VertId> v{ 0_v, 1_v, 2_v };
    auto topology0 = MeshBuilder::fromVertexTriples( v );
    auto topology1 = topology0;

    // stitch along open contour
    std::vector<std::vector<EdgeId>> c0 = { { topology0.findEdge( 1_v, 0_v ) } };
    std::vector<std::vector<EdgeId>> c1 = { { topology1.findEdge( 0_v, 1_v ) } };
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

} //namespace MR
