#include <MRMesh/MRFixSelfIntersections.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRUVSphere.h>
#include <MRMesh/MRMeshComponents.h>
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, FixSelfIntersections )
{
    Mesh mesh = makeTorusWithSelfIntersections( 1.0f, 0.2f, 32, 16 );
    EXPECT_EQ( mesh.topology.getValidFaces().count(), 1024 );

    auto intersections = SelfIntersections::getFaces( mesh, false );
    EXPECT_TRUE( intersections.has_value() );
    EXPECT_EQ( intersections->count(), 128 );

    SelfIntersections::Settings settings;
    settings.method = SelfIntersections::Settings::Method::CutAndFill;
    settings.touchIsIntersection = false;
    EXPECT_TRUE( SelfIntersections::fix( mesh, settings ).has_value() );

    EXPECT_TRUE( mesh.topology.getValidFaces().count() == 1194
              || mesh.topology.getValidFaces().count() == 1196 ); //on some macOS Arm runners in Debug mode

    intersections = SelfIntersections::getFaces( mesh, false );
    EXPECT_TRUE( intersections.has_value() );
    EXPECT_EQ( intersections->count(), 0 );
}

// A sphere with the cap above z = 0.6 moved down through its bottom crosses itself along two closed curves. The largest part between them
// is the ring bounded by the band of the sphere and by the surface that the moved cap pulls inside it. Method::SelfBoolean keeps this part
// of the closed mesh, of the mesh with a hole away from the curves, which stays as it was, and of the mesh with a hole across them;
// another part of the mesh without self-intersections is not changed.
TEST( MRMesh, FixSelfIntersectionsSelfBoolean )
{
    auto closed = makeUVSphere( 1.0f, 32, 32 );
    for ( auto& p : closed.points )
        if ( p.z > 0.6f )
            p += Vector3f( 0.1f, 0.05f, -1.7f );
    closed.invalidateCaches();
    // without the faces with centers in the ball
    auto withHole = [&] ( const Vector3f& center, float radius )
    {
        auto mesh = closed;
        FaceBitSet hole( mesh.topology.faceSize() );
        for ( auto f : mesh.topology.getValidFaces() )
            if ( ( mesh.triCenter( f ) - center ).length() < radius )
                hole.set( f );
        mesh.deleteFaces( hole );
        return mesh;
    };
    SelfIntersections::Settings settings;
    settings.method = SelfIntersections::Settings::Method::SelfBoolean;
    const float ringVolume = 0.945f;
    auto fixAndCheck = [&] ( Mesh& mesh )
    {
        EXPECT_TRUE( SelfIntersections::getFaces( mesh, false )->any() );
        EXPECT_TRUE( SelfIntersections::fix( mesh, settings ).has_value() );
        EXPECT_TRUE( SelfIntersections::getFaces( mesh, false )->none() );
        EXPECT_TRUE( mesh.topology.checkValidity() );
    };

    auto mesh = closed;
    fixAndCheck( mesh );
    EXPECT_TRUE( mesh.topology.isClosed() );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
    EXPECT_NEAR( mesh.volume(), ringVolume, 1e-3f );

    const auto away = withHole( { -1, 0, 0 }, 0.15f );
    const auto awayHole = away.topology.findHoleRepresentiveEdges();
    ASSERT_EQ( awayHole.size(), 1 );
    mesh = away;
    fixAndCheck( mesh );
    const auto fixedHole = mesh.topology.findHoleRepresentiveEdges();
    ASSERT_EQ( fixedHole.size(), 1 );
    EXPECT_EQ( mesh.topology.findLeftBdEdges().count(), away.topology.findLeftBdEdges().count() );
    EXPECT_NEAR( mesh.holePerimeter( fixedHole[0] ), away.holePerimeter( awayHole[0] ), 1e-5f );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 1 );
    EXPECT_NEAR( mesh.volume(), ringVolume, 1e-2f );

    mesh = withHole( { 0.1f, 0.6f, -0.85f }, 0.3f );
    EXPECT_FALSE( mesh.topology.isClosed() );
    fixAndCheck( mesh );
    EXPECT_NEAR( mesh.volume(), ringVolume, 1e-2f );

    // another part
    auto part = makeUVSphere( 0.5f, 16, 16 );
    part.transform( AffineXf3f::translation( { 5, 0, 0 } ) );
    mesh = part;
    mesh.addMesh( closed );
    fixAndCheck( mesh );
    EXPECT_EQ( MeshComponents::getNumComponents( mesh ), 2 );
    for ( auto f : part.topology.getValidFaces() )
    {
        ASSERT_TRUE( mesh.topology.hasFace( f ) );
        ASSERT_EQ( mesh.getTriPoints( f ), part.getTriPoints( f ) );
    }
}

} //namespace MR
