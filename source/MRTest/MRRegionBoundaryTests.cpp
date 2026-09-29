#include <MRMesh/MRRegionBoundary.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshTopology.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRId.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, findLeftBoundary)
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findLeftBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto & path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.left( e ), 0_f );
            EXPECT_NE( sphere.topology.right( e ), 0_f );
        }
    }
}

TEST( MRMesh, findRightBoundary )
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );
    FaceBitSet faces;
    faces.autoResizeSet( 0_f );
    auto paths = findRightBoundary( sphere.topology, faces );
    EXPECT_EQ( paths.size(), 1 );
    for ( const auto& path : paths )
    {
        for ( auto e : path )
        {
            EXPECT_EQ( sphere.topology.right( e ), 0_f );
            EXPECT_NE( sphere.topology.left( e ), 0_f );
        }
    }
}

TEST( MRMesh, getRegionInnerVerts )
{
    Mesh sphere = makeUVSphere( 1, 16, 16 );

    // make a hole around the upper pole
    FaceBitSet hole( sphere.topology.faceSize() );
    for ( auto f : sphere.topology.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0.8f )
            hole.set( f );
    EXPECT_GT( hole.count(), 0 );
    sphere.topology.deleteFaces( hole );

    FaceBitSet region( sphere.topology.faceSize() );
    for ( auto f : sphere.topology.getValidFaces() )
        if ( sphere.triCenter( f ).z > 0 )
            region.set( f );

    const auto verts = getRegionInnerVerts( sphere.topology, region );
    EXPECT_EQ( verts, getIncidentVerts( sphere.topology, region ) - getRegionBoundaryVerts( sphere.topology, region ) );

    // the vertices on the hole boundary are returned, unlike getInnerVerts
    const auto innerVerts = getInnerVerts( sphere.topology, region );
    EXPECT_TRUE( innerVerts.is_subset_of( verts ) );
    EXPECT_GT( verts.count(), innerVerts.count() );
}

} //namespace MR
