#include <MRMesh/MRShrinkwrap.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRMatrix3.h>
#include <gtest/gtest.h>

namespace MR
{

// large square in z=0 plane
static Mesh makePlane()
{
    Triangulation tris
    {
        { 0_v, 1_v, 2_v },
        { 0_v, 2_v, 3_v }
    };
    VertCoords ps
    {
        { -10.f, -10.f, 0.f }, // 0_v
        {  10.f, -10.f, 0.f }, // 1_v
        {  10.f,  10.f, 0.f }, // 2_v
        { -10.f,  10.f, 0.f }  // 3_v
    };
    return Mesh::fromTriangles( std::move( ps ), tris );
}

// single triangle in z=5 plane, entirely above makePlane()
static Mesh makeProbe()
{
    Triangulation tris{ { 0_v, 1_v, 2_v } };
    VertCoords ps
    {
        { 0.f, 0.f, 5.f }, // 0_v
        { 1.f, 0.f, 5.f }, // 1_v
        { 0.f, 1.f, 5.f }  // 2_v
    };
    return Mesh::fromTriangles( std::move( ps ), tris );
}

TEST( MRMesh, shrinkwrap )
{
    const auto plane = makePlane();
    auto mesh = makeProbe();
    EXPECT_TRUE( shrinkwrap( mesh, plane ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, 0.f, 1e-5f );
    // the vertices move orthogonally to the plane, keeping x and y
    EXPECT_NEAR( mesh.points[1_v].x, 1.f, 1e-5f );
    EXPECT_NEAR( mesh.points[2_v].y, 1.f, 1e-5f );
}

TEST( MRMesh, shrinkwrapOffset )
{
    const auto plane = makePlane();

    auto mesh = makeProbe();
    ShrinkwrapParameters params;
    params.offset = 2;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, 2.f, 1e-5f );

    mesh = makeProbe();
    params.offset = -2;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, -2.f, 1e-5f );
}

// a positive offset keeps the vertices on their own side of refMesh:
// the probe below the plane must stay below it, unlike with the pseudonormal direction
TEST( MRMesh, shrinkwrapOffsetKeepsSide )
{
    const auto plane = makePlane();
    Triangulation tris{ { 0_v, 1_v, 2_v } };
    VertCoords ps
    {
        { 0.f, 0.f, -5.f }, // 0_v
        { 1.f, 0.f, -5.f }, // 1_v
        { 0.f, 1.f, -5.f }  // 2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( ps ), tris );

    ShrinkwrapParameters params;
    params.offset = 2;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, -2.f, 1e-5f );
}

// the vertices lying exactly on refMesh have no side of their own, so pseudonormal is used
TEST( MRMesh, shrinkwrapOffsetOnSurface )
{
    const auto plane = makePlane();
    Triangulation tris{ { 0_v, 1_v, 2_v } };
    VertCoords ps
    {
        { 0.f, 0.f, 0.f }, // 0_v
        { 1.f, 0.f, 0.f }, // 1_v
        { 0.f, 1.f, 0.f }  // 2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( ps ), tris );

    ShrinkwrapParameters params;
    params.offset = 2;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, 2.f, 1e-5f );
}

TEST( MRMesh, shrinkwrapUpDistLimit )
{
    const auto plane = makePlane();
    auto mesh = makeProbe();
    ShrinkwrapParameters params;
    params.upDistLimitSq = 1; // the distance to the plane is 5, so no projection is found
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, 5.f, 1e-5f );
}

TEST( MRMesh, shrinkwrapRegion )
{
    const auto plane = makePlane();
    auto mesh = makeProbe();
    VertBitSet region( mesh.topology.vertSize() );
    region.set( 0_v );
    ShrinkwrapParameters params;
    params.region = &region;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    EXPECT_NEAR( mesh.points[0_v].z, 0.f, 1e-5f );
    EXPECT_NEAR( mesh.points[1_v].z, 5.f, 1e-5f );
    EXPECT_NEAR( mesh.points[2_v].z, 5.f, 1e-5f );
}

TEST( MRMesh, shrinkwrapNearestSurfacePoint )
{
    const auto plane = makePlane();
    Triangulation tris{ { 0_v, 1_v, 2_v } };
    VertCoords ps
    {
        { 12.f, 0.f, 0.f }, // 0_v beyond the plane border, the projection is on the border
        { 13.f, 0.f, 0.f }, // 1_v
        { 12.f, 1.f, 0.f }  // 2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( ps ), tris );
    EXPECT_TRUE( shrinkwrap( mesh, plane ) );
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].x, 10.f, 1e-5f );
}

TEST( MRMesh, shrinkwrapXf )
{
    const auto plane = makePlane();
    // the same probe triangle, but placed in z=5 by the transformation and not by the coordinates
    Triangulation tris{ { 0_v, 1_v, 2_v } };
    VertCoords ps
    {
        { 0.f, 0.f, 0.f }, // 0_v
        { 1.f, 0.f, 0.f }, // 1_v
        { 0.f, 1.f, 0.f }  // 2_v
    };
    auto mesh = Mesh::fromTriangles( std::move( ps ), tris );

    const AffineXf3f xf = AffineXf3f::translation( { 0.f, 0.f, 5.f } );
    ShrinkwrapParameters params;
    params.xf = &xf;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );
    // the result is returned in the coordinates of mesh, where the plane is in z=-5
    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, -5.f, 1e-5f );
}


// MeshProjectionResult::proj.point is returned in the space of refMesh when refXf is rigid
// and in world space otherwise, so shrinkwrap transfers the projection as MeshTriPoint;
// with a non-rigid refXf the proj.point based variant would return 2 0 0 for 1_v below
TEST( MRMesh, shrinkwrapNonRigidRefXf )
{
    const auto plane = makePlane();
    auto mesh = makeProbe();

    // anisotropic scaling of the reference mesh: the plane still occupies z=0 in world
    Matrix3f m;
    m.x.x = 2.f;
    const AffineXf3f refXf( m, Vector3f{} );

    PointsToMeshProjector projector;
    ShrinkwrapParameters params;
    params.refXf = &refXf;
    params.projector = &projector;
    EXPECT_TRUE( shrinkwrap( mesh, plane, params ) );

    for ( auto v : mesh.topology.getValidVerts() )
        EXPECT_NEAR( mesh.points[v].z, 0.f, 1e-5f );
    EXPECT_NEAR( mesh.points[1_v].x, 1.f, 1e-5f );
    EXPECT_NEAR( mesh.points[2_v].y, 1.f, 1e-5f );
}

} // namespace MR
