#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>
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

TEST(MRMesh, getLargestComponentArea)
{
    Triangulation t{
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v }
    };
    Mesh mesh;
    mesh.topology = MeshBuilder::fromTriangles( t );
    mesh.points.emplace_back( 0.f, 0.f, 0.f );
    mesh.points.emplace_back( 1.f, 0.f, 0.f );
    mesh.points.emplace_back( 0.f, 1.f, 0.f );
    mesh.points.emplace_back( 10.f, 0.f, 0.f );
    mesh.points.emplace_back( 12.f, 0.f, 0.f );
    mesh.points.emplace_back( 10.f, 2.f, 0.f );

    FaceBitSet largest;
    int numSmallerComponents = -1;
    ASSERT_NEAR( MeshComponents::getLargestComponentArea( mesh, MeshComponents::PerEdge, nullptr, &largest, &numSmallerComponents ), 2.0, 1e-6 );
    ASSERT_EQ( numSmallerComponents, 1 );
    ASSERT_EQ( largest.count(), 1 );
    ASSERT_TRUE( largest.test( 1_f ) );

    ASSERT_TRUE( MeshComponents::getLargestComponent( mesh, MeshComponents::PerEdge, nullptr, 1.0f, &numSmallerComponents ) == largest );
    ASSERT_EQ( numSmallerComponents, 1 );
    ASSERT_TRUE( MeshComponents::getLargestComponent( mesh, MeshComponents::PerEdge, nullptr, 3.0f, &numSmallerComponents ).none() );
    ASSERT_EQ( numSmallerComponents, 2 ); // both components are smaller than requested

    ASSERT_EQ( MeshComponents::getLargestComponentArea( Mesh{}, MeshComponents::PerEdge, nullptr, &largest, &numSmallerComponents ), 0 );
    ASSERT_TRUE( largest.none() );
    ASSERT_EQ( numSmallerComponents, 0 );
}

TEST(MRMesh, getLargestComponentVolume)
{
    using MeshComponents::VolumeSelection;
    // the small cube has volume 1 and its normals look outside, the large one has volume -8 and its normals look inside
    auto mesh = makeCube( Vector3f::diagonal( 1 ), Vector3f::diagonal( 0 ) );
    mesh.addMeshPart( makeCube( Vector3f::diagonal( 2 ), Vector3f::diagonal( 10 ) ), true );
    FaceBitSet smallCube( 24 ), largeCube( 24 );
    for ( FaceId f = 0_f; f < 12_f; ++f )
        smallCube.set( f );
    for ( FaceId f = 12_f; f < 24_f; ++f )
        largeCube.set( f );

    FaceBitSet component;
    int numSmallerComponents = -1;
    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( mesh, VolumeSelection::Abs, MeshComponents::PerEdge, nullptr, &component, &numSmallerComponents ), -8.0, 1e-5 );
    ASSERT_TRUE( component == largeCube );
    ASSERT_EQ( numSmallerComponents, 1 );

    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( mesh, VolumeSelection::Positive, MeshComponents::PerEdge, nullptr, &component, &numSmallerComponents ), 1.0, 1e-5 );
    ASSERT_TRUE( component == smallCube );
    ASSERT_EQ( numSmallerComponents, 1 );

    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( mesh, VolumeSelection::Negative, MeshComponents::PerEdge, nullptr, &component, &numSmallerComponents ), -8.0, 1e-5 );
    ASSERT_TRUE( component == largeCube );
    ASSERT_EQ( numSmallerComponents, 1 );

    // flipping the whole mesh swaps the signs of both components
    mesh.topology.flipOrientation();
    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( mesh, VolumeSelection::Positive, MeshComponents::PerEdge, nullptr, &component ), 8.0, 1e-5 );
    ASSERT_TRUE( component == largeCube );
    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( mesh, VolumeSelection::Negative, MeshComponents::PerEdge, nullptr, &component ), -1.0, 1e-5 );
    ASSERT_TRUE( component == smallCube );

    // both cubes look outside, so there is no negative component at all
    Mesh outward = makeCube( Vector3f::diagonal( 1 ), Vector3f::diagonal( 0 ) );
    outward.addMesh( makeCube( Vector3f::diagonal( 2 ), Vector3f::diagonal( 10 ) ) );
    ASSERT_NEAR( MeshComponents::getLargestComponentVolume( outward, VolumeSelection::Abs, MeshComponents::PerEdge, nullptr, &component ), 8.0, 1e-5 );
    ASSERT_TRUE( component == largeCube );
    ASSERT_EQ( MeshComponents::getLargestComponentVolume( outward, VolumeSelection::Negative, MeshComponents::PerEdge, nullptr, &component, &numSmallerComponents ), 0 );
    ASSERT_TRUE( component.none() );
    ASSERT_EQ( numSmallerComponents, 2 );

    ASSERT_EQ( MeshComponents::getLargestComponentVolume( Mesh{}, VolumeSelection::Abs, MeshComponents::PerEdge, nullptr, &component, &numSmallerComponents ), 0 );
    ASSERT_TRUE( component.none() );
    ASSERT_EQ( numSmallerComponents, 0 );
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
