#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTriangleIntersection.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

TEST( MRMesh, DegenerateTrianglesIntersect )
{
    Vector3f a{-24.5683002f,-17.7052994f,-21.3701000f};
    Vector3f b{-24.6611996f,-17.7504997f,-21.3423004f};
    Vector3f c{-24.6392994f,-17.7071991f,-21.3542995f};

    Vector3f d{-24.5401993f,-17.7504997f,-21.3390007f};
    Vector3f e{-24.5401993f,-17.7504997f,-21.3390007f};
    Vector3f f{-24.5862007f,-17.7504997f,-21.3586998f};

    bool intersection = doTrianglesIntersect(
        Vector3d{a}, Vector3d{b}, Vector3d{c},
        Vector3d{d}, Vector3d{e}, Vector3d{f} );

    // in float arithmetic this test fails unfortunately

    EXPECT_FALSE( intersection );
}

TEST( MRMesh, findSelfCollidingTriangles )
{
    Triangulation tris
    {
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    VertCoords ps
    {
        { 0.f, 0.f, 0.f }, // 0_v
        { 1.f, 1.f, 0.f }, // 1_v
        { 0.f, 1.f, 0.f }, // 2_v
        {-1.f, 0.f, 0.f }  // 3_v
    };
    Mesh mesh = Mesh::fromTriangles( std::move( ps ), tris );
    auto maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, true );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_FALSE( *maybeColl );

    mesh.points[3_v].x = 1.f;
    mesh.invalidateCaches();

    maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, false );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_FALSE( *maybeColl );

    maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, true );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_TRUE( *maybeColl );
}

} //namespace MR
