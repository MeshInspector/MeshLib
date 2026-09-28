#include <MRMesh/MROverlappingTris.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRMakeSphereMesh.h>
#include <MRMesh/MRBitSet.h>
#include <gtest/gtest.h>

namespace MR
{

TEST( MRMesh, findOverlappingTris )
{
    // triangle 0 on z=0, triangle 1 slightly above and shifted: they overlap in projection, triangle 2 is aside
    VertCoords pts = {
        { 0, 0, 0 }, { 2, 0, 0 }, { 0, 2, 0 },
        { 0.5f, 0.5f, 0.01f }, { 2.5f, 0.5f, 0.01f }, { 0.5f, 2.5f, 0.01f },
        { 3, 0, 0.01f }, { 5, 0, 0.01f }, { 3, 2, 0.01f } };
    Triangulation t = {
        { 0_v, 1_v, 2_v },
        { 3_v, 4_v, 5_v },
        { 6_v, 7_v, 8_v } };
    auto mesh = Mesh::fromTriangles( pts, t );

    FindOverlappingSettings s;
    s.maxDistSq = 0.1f * 0.1f;
    // similarly oriented triangles are not found by default
    EXPECT_EQ( findOverlappingTris( mesh, s )->count(), 0 );

    s.minNormalDot = 0.99f;
    auto res = *findOverlappingTris( mesh, s );
    EXPECT_EQ( res.count(), 2 );
    EXPECT_TRUE( res.test( 0_f ) && res.test( 1_f ) );

    // oppositely oriented triangle 1 is found by default
    std::swap( t[1_f][1], t[1_f][2] );
    mesh = Mesh::fromTriangles( pts, t );
    s.minNormalDot = FLT_MAX;
    res = *findOverlappingTris( mesh, s );
    EXPECT_EQ( res.count(), 2 );
    EXPECT_TRUE( res.test( 0_f ) && res.test( 1_f ) );

    // no false overlaps in between neighbour triangles of a smooth dense surface
    auto sphere = makeSphere( { .radius = 1, .numMeshVertices = 10000 } );
    s.maxDistSq = 0.01f * 0.01f;
    s.minNormalDot = 0.99f;
    EXPECT_EQ( findOverlappingTris( sphere, s )->count(), 0 );
}

} //namespace MR
