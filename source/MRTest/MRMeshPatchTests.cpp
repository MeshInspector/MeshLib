#include <MRMesh/MRMeshPatch.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, PatchMeshByGroups )
{
    Mesh mesh = makeCube();
    FaceBitSet faces( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto n = mesh.normal( f );
        if ( n.z > 0.9f || n.x > 0.9f )
            faces.set( f );
    }
    EXPECT_EQ( faces.count(), 4 );

    // each planar side is refilled separately, so the cube keeps its shape
    auto newFaces = patchMeshByGroups( mesh, faces );
    EXPECT_EQ( newFaces.count(), 4 );
    EXPECT_EQ( mesh.topology.numValidFaces(), 12 );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_NEAR( mesh.volume(), 1.0, 1e-5 );

    // fine subdivision splits the edge between the groups: all faces of the top and the side must still be replaced
    mesh = makeCube();
    FillHoleNicelySettings fine;
    fine.subdivideSettings.maxEdgeLen = 0.2f;
    fine.smoothCurvature = false;
    newFaces = patchMeshByGroups( mesh, faces, 0.5f, fine );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_NEAR( mesh.volume(), 1.0, 1e-5 );
    for ( auto f : mesh.topology.getValidFaces() )
    {
        const auto n = mesh.normal( f );
        const auto c = mesh.triCenter( f );
        const bool topOrSide = ( n.z > 0.999f && c.z > 0.4999f ) || ( n.x > 0.999f && c.x > 0.4999f );
        EXPECT_EQ( topOrSide, newFaces.test( f ) );
    }
}

} //namespace MR
