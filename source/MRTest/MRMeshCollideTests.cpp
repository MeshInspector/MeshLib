#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRGTest.h>

namespace MR
{

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
    // co-planar triangles with same orientation
    auto maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, true );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_FALSE( *maybeColl );

    // not co-planar triangles
    mesh.points[3_v] = Vector3f{ 1.f, 0.f, 1.f };
    mesh.invalidateCaches();
    maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, true );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_FALSE( *maybeColl );

    // co-planar triangles with opposite orientation
    mesh.points[3_v] = Vector3f{ 1.f, 0.f, 0.f };
    mesh.invalidateCaches();

    maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, false );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_FALSE( *maybeColl );

    maybeColl = findSelfCollidingTriangles( mesh, nullptr, ProgressCallback{}, nullptr, true );
    EXPECT_TRUE( maybeColl.has_value() );
    EXPECT_TRUE( *maybeColl );
}

} //namespace MR
