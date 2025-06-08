#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRGTest.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRCube.h>

namespace MR
{

TEST(MRMesh, getAllComponentsEdges)
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

    EdgeId e12 = mesh.topology.findEdge( 1_v, 2_v );
    EdgeId e30 = mesh.topology.findEdge( 3_v, 0_v );

    EdgeBitSet ebs( 10 );
    ebs.set( e12 );
    ebs.set( e30 );
    auto comp = MeshComponents::getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 2 );
    ASSERT_EQ( comp[0].count(), 1 );
    ASSERT_EQ( comp[1].count(), 1 );

    ebs.set( e12.sym() );
    ebs.set( e30.sym() );
    comp = MeshComponents::getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 2 );
    ASSERT_EQ( comp[0].count(), 2 );
    ASSERT_EQ( comp[1].count(), 2 );

    ebs.set( mesh.topology.findEdge( 0_v, 1_v ) );
    comp = MeshComponents::getAllComponentsEdges( mesh, ebs );
    ASSERT_EQ( comp.size(), 1 );
    ASSERT_EQ( comp[0].count(), 5 );
}

TEST(MRMesh, getLargestComponentVerts)
{
    auto mesh = makeCube();
    {
        auto l = MeshComponents::getLargestComponentVerts( mesh );
        ASSERT_EQ( l.size(), 8 );
        ASSERT_EQ( l.count(), 8 );
    }
    {
        VertBitSet region( 8 );
        region.set( 1_v );
        region.set( 2_v );
        region.set( 7_v );
        auto l = MeshComponents::getLargestComponentVerts( mesh, &region );
        ASSERT_EQ( l.size(), 8 );
        ASSERT_EQ( l.count(), 2 );
        ASSERT_TRUE( l.test( 1_v ) );
        ASSERT_TRUE( l.test( 2_v ) );
    }
}

} //namespace MR
