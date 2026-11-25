#include <MRMesh/MRMeshMeshDistance.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMeshBuilder.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include "MRMesh/MRCube.h"
#include "MRMesh/MRConstants.h"
#include <MRMesh/MRAffineXf3.h>
#include <gtest/gtest.h>

namespace MR
{

TEST(MRMesh, MeshDistance)
{
    Mesh sphere1 = makeUVSphere( 1, 8, 8 );

    auto d11 = findDistance( sphere1, sphere1, nullptr, FLT_MAX );
    EXPECT_EQ( d11.distSq, 0 );

    auto zShift = AffineXf3f::translation( Vector3f( 0, 0, 3 ) );
    auto d1z = findDistance( sphere1, sphere1, &zShift, FLT_MAX );
    EXPECT_EQ( d1z.distSq, 1 );

    Mesh sphere2 = makeUVSphere( 2, 8, 8 );

    auto d12 = findDistance( sphere1, sphere2, nullptr, FLT_MAX );
    float dist12 = std::sqrt( d12.distSq );
    EXPECT_TRUE( dist12 > 0.9f && dist12 < 1.0f );
}

TEST( MRMesh, MeshMeshClosestPoint )
{
    auto cube1 = makeCube();
    auto cube2 = makeCube();
    AffineXf3f b2a = AffineXf3f::translation( Vector3f( 0.8f, 0.8f, 0.0f ) ) *
        AffineXf3f::linear( Matrix3f::rotationFromEuler( Vector3f( PI_F * 0.25f, 0.0f, -PI_F * 0.25f ) ) );

    auto sd = findSignedDistance( cube1, cube2, &b2a );
    EXPECT_EQ( sd.signedDist, 0.0f );

    auto usd = findDistance( cube1, cube2, &b2a );
    EXPECT_EQ( usd.distSq, 0.0f );
    EXPECT_TRUE( findProjection( usd.a.proj.point, cube1 ).distSq < 1e-6f );
    EXPECT_TRUE( findProjection( usd.b.proj.point, cube2 ).distSq < 1e-6f );
}

TEST( MRMesh, HausdorffDistance )
{
    // all vertices of the triangle are on the cube, and the center of the cube inside the triangle is the farthest from it
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    const Mesh tri = Mesh::fromTriangles( { Vector3f( 0, 0, 0.5f ), Vector3f( 1, 0, 0.5f ), Vector3f( 0.3f, 1, 0.5f ) }, t );
    const float tol = 1e-3f;
    const Vector3f center( 0.5f, 0.5f, 0.5f );

    EXPECT_EQ( findMaxDistanceSqOneWay( cube, tri ), 0 );
    auto res = findHausdorffDistanceOneWay( cube, tri, tol );
    EXPECT_LE( std::sqrt( res.distSq ), 0.5f );
    EXPECT_GE( std::sqrt( res.distSq ), 0.5f - tol );
    ASSERT_TRUE( res.a.valid() && res.b.valid() );
    EXPECT_LE( ( res.b.proj.point - center ).length(), 2 * tol );
    EXPECT_NEAR( ( res.a.proj.point - res.b.proj.point ).lengthSq(), res.distSq, 1e-6f );
    EXPECT_LE( ( tri.triPoint( res.b.mtp( tri.topology ) ) - res.b.proj.point ).length(), 1e-6f );
    EXPECT_LE( ( cube.triPoint( res.a.mtp( cube.topology ) ) - res.a.proj.point ).length(), 1e-6f );

    // the farthest points of the cube from the triangle are its two vertices with x=1, y=1
    const auto cubeMaxSq = 0.25f + 0.49f / 1.49f;
    res = findHausdorffDistanceOneWay( tri, cube, tol );
    EXPECT_NEAR( res.distSq, cubeMaxSq, 1e-6f );
    EXPECT_EQ( res.b.proj.point.x, 1 );
    EXPECT_EQ( res.b.proj.point.y, 1 );
    EXPECT_EQ( res.a.proj.point.z, 0.5f );

    // the farthest point is in the second mesh
    res = findHausdorffDistance( cube, tri, tol );
    EXPECT_NEAR( res.distSq, cubeMaxSq, 1e-6f );
    EXPECT_EQ( res.a.proj.point.x, 1 );
    EXPECT_EQ( res.b.proj.point.z, 0.5f );

    // the triangle moved away, and the transformation returning it back
    const auto xf = AffineXf3f::translation( Vector3f( 3, 2, 1 ) ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1, 2, 3 ), 1.0f ) );
    const auto xfInv = xf.inverse();
    Mesh movedTri = tri;
    movedTri.transform( xfInv );
    res = findHausdorffDistanceOneWay( cube, movedTri, tol, &xf );
    EXPECT_LE( std::sqrt( res.distSq ), 0.5f + 1e-5f );
    EXPECT_GE( std::sqrt( res.distSq ), 0.5f - tol );
    // the point of B is returned in its own space
    EXPECT_LE( ( xf( res.b.proj.point ) - center ).length(), 2 * tol );
    EXPECT_NEAR( findHausdorffDistance( cube, movedTri, tol, &xf ).distSq, cubeMaxSq, 1e-5f );
    EXPECT_NEAR( findHausdorffDistance( movedTri, cube, tol, &xfInv ).distSq, cubeMaxSq, 1e-5f );

    res = findHausdorffDistanceOneWay( cube, tri, tol, nullptr, 0.01f );
    EXPECT_EQ( res.distSq, 0.01f );
    EXPECT_FALSE( res.a.valid() );
    EXPECT_TRUE( res.b.valid() );

    EXPECT_EQ( findHausdorffDistance( cube, cube, tol ).distSq, 0 );

    // the limit is reached in the second direction only, where the vertices of the cube are too far
    res = findHausdorffDistance( cube, tri, tol, nullptr, 0.01f );
    EXPECT_EQ( res.distSq, 0.01f );
    EXPECT_TRUE( res.a.valid() );
    EXPECT_FALSE( res.b.valid() );

    // without positive tolerance only the vertices are considered
    for ( float badTol : { 0.f, -1.f, std::numeric_limits<float>::quiet_NaN() } )
    {
        EXPECT_EQ( findHausdorffDistanceOneWay( cube, tri, badTol ).distSq, 0 );
        EXPECT_EQ( findHausdorffDistance( cube, tri, badTol ).distSq, findMaxDistanceSq( cube, tri ) );
    }
}

TEST( MRMesh, HausdorffDistanceSlivers )
{
    // the farthest point of the long thin strip is in the middle between two small triangles;
    // away from zero the short edges of the strip cannot be halved in floats many times
    Triangulation tb;
    tb.push_back( { 0_v, 1_v, 2_v } );
    tb.push_back( { 0_v, 2_v, 3_v } );
    const Mesh strip = Mesh::fromTriangles( { Vector3f( 0, 1, 0 ), Vector3f( 1, 1, 0 ), Vector3f( 1, 1 + 1e-5f, 0 ), Vector3f( 0, 1 + 1e-5f, 0 ) }, tb );
    Triangulation ta;
    ta.push_back( { 0_v, 1_v, 2_v } );
    ta.push_back( { 3_v, 4_v, 5_v } );
    const Mesh ends = Mesh::fromTriangles( {
        Vector3f( -0.1f, 1, 0 ), Vector3f( -0.1f, 1.001f, 0 ), Vector3f( -0.101f, 1, 0 ),
        Vector3f( 1.2f, 1, 0 ), Vector3f( 1.201f, 1, 0 ), Vector3f( 1.2f, 1.001f, 0 ) }, ta );

    for ( float tol : { 1e-3f, 1e-5f } )
    {
        const auto res = findHausdorffDistanceOneWay( ends, strip, tol );
        EXPECT_LE( std::sqrt( res.distSq ), 0.65f + 1e-6f );
        EXPECT_GE( std::sqrt( res.distSq ), 0.65f - tol - 1e-6f );
    }
}

TEST( MRMesh, HausdorffDistanceRegions )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    t.push_back( { 3_v, 4_v, 5_v } );
    // the second triangle is far away
    const Mesh tris = Mesh::fromTriangles( { Vector3f( 0, 0, 0.5f ), Vector3f( 1, 0, 0.5f ), Vector3f( 0.3f, 1, 0.5f ),
        Vector3f( 5, 0, 0 ), Vector3f( 6, 0, 0 ), Vector3f( 5, 1, 0 ) }, t );
    const float tol = 1e-3f;

    auto cubeFaces = [&]( int axis, float value )
    {
        FaceBitSet res( cube.topology.faceSize() );
        for ( auto f : cube.topology.getValidFaces() )
        {
            const auto ps = cube.getTriPoints( f );
            if ( ps[0][axis] == value && ps[1][axis] == value && ps[2][axis] == value )
                res.set( f );
        }
        return res;
    };
    const auto bottom = cubeFaces( 2, 0 );
    const auto front = cubeFaces( 1, 0 );
    ASSERT_EQ( bottom.count(), 2 );
    ASSERT_EQ( front.count(), 2 );
    FaceBitSet firstTri( 2 );
    firstTri.set( 0_f );

    // region in B: the far triangle is ignored
    EXPECT_GT( findHausdorffDistanceOneWay( cube, tris, tol ).distSq, 16 );
    auto res = findHausdorffDistanceOneWay( cube, { tris, &firstTri }, tol );
    EXPECT_LE( std::sqrt( res.distSq ), 0.5f );
    EXPECT_GE( std::sqrt( res.distSq ), 0.5f - tol );
    EXPECT_EQ( res.b.proj.face, 0_f );

    // region in A: only the bottom of the cube, and all points of the first triangle are 0.5 away from it
    res = findHausdorffDistanceOneWay( { cube, &bottom }, { tris, &firstTri }, tol );
    EXPECT_EQ( res.distSq, 0.25f );
    EXPECT_TRUE( bottom.test( res.a.proj.face ) );

    // the first triangle has an edge in the front side of the cube, the farthest points of that side are 0.5 away from it
    res = findHausdorffDistanceOneWay( { tris, &firstTri }, { cube, &front }, tol );
    EXPECT_EQ( res.distSq, 0.25f );
    EXPECT_TRUE( front.test( res.b.proj.face ) );
    // and the third vertex of the triangle is 1 away from the front side
    EXPECT_EQ( findHausdorffDistance( { tris, &firstTri }, { cube, &front }, tol ).distSq, 1 );
    EXPECT_EQ( findHausdorffDistance( { cube, &front }, { tris, &firstTri }, tol ).distSq, 1 );
}

TEST( MRMesh, HausdorffDistanceInsideTriangleXf )
{
    // the farthest point is on the long side of the strip in the middle between two small triangles, not in a vertex
    Triangulation tb;
    tb.push_back( { 0_v, 1_v, 2_v } );
    tb.push_back( { 0_v, 2_v, 3_v } );
    const Mesh strip = Mesh::fromTriangles( { Vector3f( 0, 0, 0 ), Vector3f( 1, 0, 0 ), Vector3f( 1, 0.3f, 0 ), Vector3f( 0, 0.3f, 0 ) }, tb );
    Triangulation ta;
    ta.push_back( { 0_v, 1_v, 2_v } );
    ta.push_back( { 3_v, 4_v, 5_v } );
    const Mesh ends = Mesh::fromTriangles( {
        Vector3f( -0.1f, 0, 0 ), Vector3f( -0.1f, 0.001f, 0 ), Vector3f( -0.101f, 0, 0 ),
        Vector3f( 1.2f, 0, 0 ), Vector3f( 1.201f, 0, 0 ), Vector3f( 1.2f, 0.001f, 0 ) }, ta );
    const float tol = 1e-3f;
    const Vector3f farthest( 0.55f, 0.3f, 0 );
    const float dist = std::sqrt( sqr( 0.65f ) + sqr( 0.299f ) );

    const auto xf = AffineXf3f::translation( Vector3f( 3, 2, 1 ) ) * AffineXf3f::linear( Matrix3f::rotation( Vector3f( 1, 2, 3 ), 1.0f ) );
    const auto xfInv = xf.inverse();
    Mesh movedStrip = strip;
    movedStrip.transform( xfInv );

    auto check = [&]( const PointOnFaceBary& onStrip, const PointOnFaceBary& onEnds, float distSq )
    {
        EXPECT_LE( std::sqrt( distSq ), dist + 1e-5f );
        EXPECT_GE( std::sqrt( distSq ), dist - tol );
        ASSERT_TRUE( onStrip.valid() && onEnds.valid() );
        // each point is in the space of its own mesh
        EXPECT_LE( ( xf( onStrip.proj.point ) - farthest ).length(), 5 * tol );
        EXPECT_LE( ( movedStrip.triPoint( onStrip.mtp( movedStrip.topology ) ) - onStrip.proj.point ).length(), 1e-6f );
        EXPECT_LE( ( ends.triPoint( onEnds.mtp( ends.topology ) ) - onEnds.proj.point ).length(), 1e-6f );
        EXPECT_NEAR( ( xf( onStrip.proj.point ) - onEnds.proj.point ).length(), std::sqrt( distSq ), 1e-5f );
    };
    auto res = findHausdorffDistanceOneWay( ends, movedStrip, tol, &xf );
    check( res.b, res.a, res.distSq );
    res = findHausdorffDistance( ends, movedStrip, tol, &xf );
    check( res.b, res.a, res.distSq );
    // the farthest point is found in the second direction
    res = findHausdorffDistance( movedStrip, ends, tol, &xfInv );
    check( res.a, res.b, res.distSq );
}

TEST( MRMesh, HausdorffDistanceDegenerateTriangles )
{
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    const float tol = 1e-3f;
    Triangulation t;
    t.push_back( { 0_v, 1_v, 2_v } );
    // both triangles are segments through the center of the cube: with zero area, and with two coinciding vertices
    for ( const auto& third : { Vector3f( 0.25f, 0.5f, 0.5f ), Vector3f( 0, 0.5f, 0.5f ) } )
    {
        const Mesh seg = Mesh::fromTriangles( { Vector3f( 0, 0.5f, 0.5f ), Vector3f( 1, 0.5f, 0.5f ), third }, t );
        const auto res = findHausdorffDistanceOneWay( cube, seg, tol );
        EXPECT_LE( std::sqrt( res.distSq ), 0.5f );
        EXPECT_GE( std::sqrt( res.distSq ), 0.5f - tol );
        EXPECT_LE( ( res.b.proj.point - Vector3f::diagonal( 0.5f ) ).length(), 2 * tol );
    }
}

TEST( MRMesh, HausdorffDistanceRounding )
{
    // two triangulations of the same tilted square far from the origin, with the tolerance below the resolution of floats there
    const Vector3f c( 100, 100, 100 );
    const Vector3f u = 0.0005f / 3 * Vector3f( 1, 2, 2 );
    const Vector3f v = 0.0005f / 3 * Vector3f( 2, 1, -2 );
    const VertCoords ps{ c - u - v, c + u - v, c + u + v, c - u + v };
    Triangulation t1;
    t1.push_back( { 0_v, 1_v, 2_v } );
    t1.push_back( { 0_v, 2_v, 3_v } );
    Triangulation t2;
    t2.push_back( { 0_v, 1_v, 3_v } );
    t2.push_back( { 1_v, 2_v, 3_v } );
    const Mesh m1 = Mesh::fromTriangles( ps, t1 );
    const Mesh m2 = Mesh::fromTriangles( ps, t2 );

    const auto res = findHausdorffDistanceOneWay( m1, m2, 1e-6f );
    const float ulp = 7.63e-6f; // of 100
    // the returned distance is the one from the returned point, and rounding errors do not accumulate in that point
    EXPECT_NEAR( std::sqrt( res.distSq ), std::sqrt( findProjection( res.b.proj.point, m1 ).distSq ), ulp );
    EXPECT_LE( std::sqrt( res.distSq ), 3 * ulp );
}

TEST( MRMesh, MaxDistancePoints )
{
    // the vertices of the tall box with z=2 are the farthest from the cube, and all vertices of the cube are on the tall box
    const Mesh cube = makeCube( Vector3f::diagonal( 1 ), Vector3f() );
    const Mesh box = makeCube( Vector3f( 1, 1, 2 ), Vector3f() );

    auto res = findMaxDistanceOneWay( cube, box );
    EXPECT_EQ( res.distSq, 1 );
    EXPECT_EQ( res.distSq, findMaxDistanceSqOneWay( cube, box ) );
    ASSERT_TRUE( res.a.valid() && res.b.valid() );
    EXPECT_EQ( res.b.proj.point.z, 2 );
    EXPECT_EQ( res.a.proj.point, res.b.proj.point - Vector3f( 0, 0, 1 ) );
    EXPECT_EQ( box.triPoint( res.b.mtp( box.topology ) ), res.b.proj.point );

    EXPECT_EQ( findMaxDistanceOneWay( box, cube ).distSq, 0 );

    // the farthest vertex is in the first mesh
    res = findMaxDistance( box, cube );
    EXPECT_EQ( res.distSq, 1 );
    EXPECT_EQ( res.distSq, findMaxDistanceSq( box, cube ) );
    EXPECT_EQ( res.a.proj.point.z, 2 );
    EXPECT_EQ( res.b.proj.point.z, 1 );

    // the point of B is returned in its own space
    const auto xf = AffineXf3f::translation( Vector3f( 0, 0, 3 ) );
    res = findMaxDistance( cube, box, &xf );
    EXPECT_EQ( res.distSq, 16 );
    EXPECT_EQ( res.a.proj.point.z, 1 );
    EXPECT_EQ( res.b.proj.point.z, 2 );

    res = findMaxDistanceOneWay( cube, box, nullptr, 0.25f );
    EXPECT_EQ( res.distSq, 0.25f );
    EXPECT_FALSE( res.a.valid() );
    EXPECT_TRUE( res.b.valid() );
}

} //namespace MR
