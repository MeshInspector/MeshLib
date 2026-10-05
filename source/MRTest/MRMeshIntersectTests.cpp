#include <gtest/gtest.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshIntersect.h>
#include <MRMesh/MRLine3.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MREdgeIterator.h>

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

TEST( MRMesh, MeshIntersectAllPrecise )
{
    auto countHits = [] ( const Mesh & mesh, const Line3d & line )
    {
        int numHits = 0;
        rayMeshIntersectAll( mesh, line, [&numHits] ( const MeshIntersectionResult & ) { ++numHits; return true; }, -DBL_MAX, DBL_MAX );
        return numHits;
    };

    // the face of the cube at x=0 lies in the plane of the bounding box, and the segment's end clipped by the box must not land on it
    const auto cube1 = makeCube( Vector3f::diagonal( 1 ), Vector3f::diagonal( -1 ) );
    EXPECT_EQ( countHits( cube1, Line3d( Vector3d( -2, -0.3, -0.6 ), Vector3d::plusX() ) ), 2 );

    // the ray passes the vertex in the center of the face x=-0.5 closer than the step of integer coordinates,
    // and the triangle below the vertex must not be culled
    auto cube2 = makeCube();
    for ( auto ue : undirectedEdges( cube2.topology ) )
    {
        const auto a = cube2.orgPnt( ue );
        const auto b = cube2.destPnt( ue );
        if ( a.x == -0.5f && b.x == -0.5f && ( a - b ).length() > 1.1f )
        {
            cube2.splitEdge( ue );
            break;
        }
    }
    EXPECT_EQ( countHits( cube2, Line3d( Vector3d( -1, 0, 1e-12 ), Vector3d::plusX() ) ), 2 );
}

} //namespace MR
