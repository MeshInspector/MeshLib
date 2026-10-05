#include <gtest/gtest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine3.h>

namespace MR
{

TEST(MRMesh, MeshIntersect) 
{
    Mesh sphere = makeUVSphere( 1, 8, 8 );

    std::vector<MeshIntersectionResult> allFound;
    auto callback = [&allFound]( const MeshIntersectionResult & found ) -> bool
    {
        allFound.push_back( found );
        return true;
    };

    Vector3f d{ 1, 2, 3 };
    rayMeshIntersectAll( sphere, { 2.0f * d, -d.normalized() }, callback );
    ASSERT_EQ( allFound.size(), 2 );
    for ( const auto & found : allFound )
    {
        ASSERT_NEAR( found.proj.point.length(), 1.0f, 0.05f ); //our sphere is very approximate
    }

    const auto isect1 = rayMeshIntersect( sphere, { Vector3f{ +0.1f, 0.f, 0.f }, Vector3f::plusX() }, -FLT_MAX, +FLT_MAX );
    EXPECT_TRUE( isect1 );
    EXPECT_NEAR( isect1.distanceAlongLine, 0.9f, 0.05f );
    EXPECT_NEAR( isect1.proj.point.x, +1.f, 0.05f );

    const auto isect2 = rayMeshIntersect( sphere, { Vector3f{ -0.1f, 0.f, 0.f }, Vector3f::plusX() }, -FLT_MAX, +FLT_MAX );
    EXPECT_TRUE( isect2 );
    EXPECT_NEAR( isect2.distanceAlongLine, -0.9f, 0.05f );
    EXPECT_NEAR( isect2.proj.point.x, -1.f, 0.05f );
}

TEST(MRMesh, MeshIntersectAllPrecise)
{
    // rays through the vertices and along the diagonal edges of the cube's triangles
    Mesh cube = makeCube();
    for ( const auto & line : {
        Line3d( Vector3d( -1, 0, 0 ), Vector3d( 1, 0, 0 ) ),
        Line3d( Vector3d( -1, -1, -1 ), Vector3d( 1, 1, 1 ) ),
        Line3d( Vector3d( 0, -1, 0 ), Vector3d( 0, 1, 0 ) ) } )
    {
        int count = 0;
        rayMeshIntersectAllPrecise( cube, line, [&count] ( const MeshIntersectionResult & ) { ++count; return true; }, -DBL_MAX, DBL_MAX );
        EXPECT_EQ( count % 2, 0 );
    }
}

} //namespace MR
