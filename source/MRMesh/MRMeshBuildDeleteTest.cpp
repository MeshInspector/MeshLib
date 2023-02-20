#include "MRMeshBuilder.h"
#include "MRMeshTopology.h"
#include "MRBitSet.h"
#include "MRExpandShrink.h"
#include "MRRegionBoundary.h"
#include "MRGTest.h"

namespace MR
{

TEST(MRMesh, BuildTri) 
{
    Triangulation t{ { 0_v, 1_v, 2_v } };
    auto topology = MeshBuilder::fromTriangles( t );

    EXPECT_EQ( topology.numValidVerts(), 3 );
    EXPECT_EQ( topology.numValidFaces(), 1 );

    auto bdEdges = topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 1 );
    EXPECT_FALSE( topology.left( bdEdges[0] ).valid() );
}

TEST(MRMesh, fromFaceSoup) 
{
    std::vector<VertId> vs
    { 
        0_v, 1_v, 2_v, 3_v, 4_v,
        1_v, 0_v, 5_v, 6_v,
        1_v, 6_v, 2_v
    };
    std::vector<MeshBuilder::FaceRecord> faceRecs
    {
        { 0_f, 0, 5 },
        { 1_f, 5, 9 },
        { 2_f, 9, 12 }
    };
    auto topology = fromFaceSoup( vs, faceRecs );

    EXPECT_EQ( topology.numValidVerts(), 7 );
    EXPECT_EQ( topology.numValidFaces(), 3 );
    EXPECT_EQ( topology.getFaceDegree( 0_f ), 5 );
    EXPECT_EQ( topology.getFaceDegree( 1_f ), 4 );
    EXPECT_EQ( topology.getFaceDegree( 2_f ), 3 );

    auto bdEdges = topology.findHoleRepresentiveEdges();
    EXPECT_EQ( bdEdges.size(), 1 );
    EXPECT_FALSE( topology.left( bdEdges[0] ).valid() );
}

TEST(MRMesh, BuildQuadDelete) 
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    auto topology = MeshBuilder::fromTriangles( t );

    EXPECT_EQ( topology.numValidVerts(), 4 );
    EXPECT_EQ( topology.getValidVerts().count(), 4 );
    EXPECT_EQ( topology.numValidFaces(), 2 );
    EXPECT_EQ( topology.getValidFaces().count(), 2 );
    EXPECT_EQ( topology.computeNotLoneUndirectedEdges(), 5 );

    auto e02 = topology.findEdge( VertId( 0 ), VertId( 2 ) );
    EXPECT_TRUE( e02.valid() );
    EXPECT_TRUE( topology.left( e02 ).valid() );
    EXPECT_TRUE( topology.right( e02 ).valid() );
    EXPECT_NE( topology.left( e02 ), topology.right( e02 ) );

    auto e13 = topology.findEdge( VertId( 1 ), VertId( 3 ) );
    EXPECT_FALSE( e13.valid() );

    FaceBitSet faces( 2 );
    faces.set( FaceId( 0 ) );
    EXPECT_EQ( getIncidentVerts( topology, faces ).count(), 3 );
    EXPECT_EQ( getInnerVerts( topology, faces ).count(), 0 );

    FaceBitSet fs = faces;
    expand( topology, fs );
    EXPECT_EQ( fs.count(), 2 );

    fs = faces;
    shrink( topology, fs );
    EXPECT_EQ( fs.count(), 0 );

    VertBitSet verts( 4 );
    verts.set( VertId( 1 ) );
    EXPECT_EQ( getIncidentFaces( topology, verts ).count(), 1 );
    EXPECT_EQ( getInnerFaces( topology, verts ).count(), 0 );

    // now check deletion
    topology.deleteFace( 1_f );
    EXPECT_EQ( topology.numValidVerts(), 3 );
    EXPECT_EQ( topology.getValidVerts().count(), 3 );
    EXPECT_EQ( topology.numValidFaces(), 1 );
    EXPECT_EQ( topology.getValidFaces().count(), 1 );
    EXPECT_EQ( topology.computeNotLoneUndirectedEdges(), 3 );

    topology.deleteFace( 0_f );
    EXPECT_EQ( topology.numValidVerts(), 0 );
    EXPECT_EQ( topology.getValidVerts().count(), 0 );
    EXPECT_EQ( topology.numValidFaces(), 0 );
    EXPECT_EQ( topology.getValidVerts().count(), 0 );
    EXPECT_EQ( topology.computeNotLoneUndirectedEdges(), 0 );
}

TEST(MRMesh, FlipEdge) 
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    auto topology = MeshBuilder::fromTriangles( t );

    auto e = topology.findEdge( VertId{0}, VertId{2} );
    EXPECT_TRUE( e.valid() );
    auto l = topology.left( e );
    auto r = topology.right( e );
    EXPECT_EQ( topology.org( e ), VertId{0} );
    EXPECT_EQ( topology.dest( e ), VertId{2} );
    EXPECT_TRUE( topology.isLeftTri( e ) );
    EXPECT_TRUE( topology.isLeftTri( e.sym() ) );

    topology.flipEdge( e );
    EXPECT_EQ( topology.left( e ), l );
    EXPECT_EQ( topology.right( e ), r );
    EXPECT_EQ( topology.org( e ), VertId{1} );
    EXPECT_EQ( topology.dest( e ), VertId{3} );
    EXPECT_TRUE( topology.isLeftTri( e ) );
    EXPECT_TRUE( topology.isLeftTri( e.sym() ) );
    EXPECT_NE( topology.edgeWithOrg( VertId{0} ), e );
    EXPECT_NE( topology.edgeWithOrg( VertId{2} ), e.sym() );
}

} //namespace MR
