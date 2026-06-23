#include <MRMesh/MRFillContours2D.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRVector3.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshTrimWithPlane.h>
#include <MRMesh/MRPlane3.h>
#include <gtest/gtest.h>
#include <limits>

namespace MR
{

TEST( MRMesh, fillContours2D )
{
    Mesh sphereBig = makeUVSphere( 1.0f, 32, 32 );
    Mesh sphereSmall = makeUVSphere( 0.7f, 16, 16 );

    sphereSmall.topology.flipOrientation();
    sphereBig.addMesh( sphereSmall );

    trimWithPlane( sphereBig, TrimWithPlaneParams{ .plane = Plane3f::fromDirAndPt( Vector3f::plusZ(), Vector3f() ) } );
    sphereBig.pack();

    auto firstNewFace = sphereBig.topology.lastValidFace() + 1;
    auto v = fillContours2D( sphereBig, sphereBig.topology.findHoleRepresentiveEdges() );
    EXPECT_TRUE( v.has_value() );
    for ( FaceId f = firstNewFace; f <= sphereBig.topology.lastValidFace(); ++f )
    {
        EXPECT_TRUE( std::abs( dot( sphereBig.dirDblArea( f ).normalized(), Vector3f::minusZ() ) - 1.0f ) < std::numeric_limits<float>::epsilon() );
    }
}

// Characterizes fillContours2D on a hole lying in a plane tilted off all axes (so the projection /
// dominant-axis path is exercised). Pins the observable result so the upcoming mesh-space rewrite
// cannot change it blindly: the hole must close watertight, reuse the boundary vertices in place
// (no drift or duplication), and the patch must lie in the cut plane.
TEST( MRMesh, fillContours2DTiltedPlane )
{
    Mesh mesh = makeUVSphere( 1.0f, 32, 32 );
    const Vector3f normal = Vector3f( 1.f, 2.f, 3.f ).normalized();
    trimWithPlane( mesh, TrimWithPlaneParams{ .plane = Plane3f::fromDirAndPt( normal, Vector3f() ) } );
    mesh.pack();

    ASSERT_EQ( mesh.topology.findHoleRepresentiveEdges().size(), size_t( 1 ) );
    const int vertsBefore = mesh.topology.numValidVerts();
    const FaceId firstNewFace = mesh.topology.lastValidFace() + 1;

    const auto res = fillContours2D( mesh, mesh.topology.findHoleRepresentiveEdges() );
    EXPECT_TRUE( res.has_value() );

    // hole closed watertight, with the boundary vertices reused in place (no new / duplicated verts)
    EXPECT_TRUE( mesh.topology.findHoleRepresentiveEdges().empty() );
    EXPECT_EQ( mesh.topology.numValidVerts(), vertsBefore );

    // every patch face lies in the cut plane, oriented toward the removed -normal side (this also rules
    // out degenerate faces, whose normal would not align). Triangle quality is intentionally not pinned:
    // the current fill emits slivers, and the exact triangulation is not a stable invariant to guard.
    for ( FaceId f = firstNewFace; f <= mesh.topology.lastValidFace(); ++f )
        EXPECT_GT( dot( mesh.normal( f ), -normal ), 0.99f );
}

}
