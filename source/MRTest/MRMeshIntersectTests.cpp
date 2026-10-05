#include <gtest/gtest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine3.h>
#include <MRMesh/MRCube.h>

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

TEST(MRMesh, MeshIntersectAllDistanceAlongLine)
{
    // non-unit direction: faces x=-0.5 and x=+0.5 are at t=0.25 and t=0.75
    Mesh cube = makeCube();
    const Vector3d p( -1, 0.1, 0.2 ), d( 2, 0, 0 );
    for ( bool useDouble : { false, true } )
    {
        std::vector<float> ts;
        auto callback = [&ts] ( const MeshIntersectionResult & found ) { ts.push_back( found.distanceAlongLine ); return true; };
        if ( useDouble )
            rayMeshIntersectAll( cube, Line3d( p, d ), callback, 0.0, 1.0 );
        else
            rayMeshIntersectAll( cube, Line3f( Vector3f( p ), Vector3f( d ) ), callback, 0.0f, 1.0f );
        std::sort( ts.begin(), ts.end() );
        ASSERT_EQ( ts.size(), 2 );
        EXPECT_NEAR( ts[0], 0.25f, 1e-6f );
        EXPECT_NEAR( ts[1], 0.75f, 1e-6f );
    }
}

} //namespace MR
