#include <MRMesh/MR2DContoursTriangulation.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRContour.h>
#include <MRMesh/MRVector2.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, PlanarTriangulation )
{
    // Create a quadrangle with three points on a straight line
    Contour2f cont;
    cont.push_back( Vector2f( 1.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 0.f ) );
    cont.push_back( Vector2f( 0.f, 1.f ) );
    cont.push_back( Vector2f( 0.f, 2.f ) );
    cont.push_back( Vector2f( 1.f, 0.f ) );

    Mesh mesh = PlanarTriangulation::triangulateContours( { cont } );
    mesh.pack();
    EXPECT_TRUE( mesh.topology.lastValidFace() == 1_f );
    // Must not contain degenerate faces
    EXPECT_TRUE( mesh.triangleAspectRatio( 0_f ) < 10.0f );
    EXPECT_TRUE( mesh.triangleAspectRatio( 1_f ) < 10.0f );
}

} //namespace MR
