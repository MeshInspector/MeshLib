#include <MRMesh/MRFillContours2D.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMeshTrimWithPlane.h>

#include <gtest/gtest.h>

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

} // namespace MR
