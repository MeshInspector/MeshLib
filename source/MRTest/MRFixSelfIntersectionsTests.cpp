#include <MRMesh/MRFixSelfIntersections.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRBitSet.h>
#include "MRGTest.h"

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

} //namespace MR
