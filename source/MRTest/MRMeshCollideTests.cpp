#include <MRMesh/MRMeshCollide.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRTorus.h>
#include <MRMesh/MRAffineXf3.h>
#include <MRMesh/MRBitSet.h>
#include <algorithm>
#include <array>
#include <gtest/gtest.h>

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

TEST( MRMesh, findCollidingTrianglesFirstOnly )
{
    const Mesh a = makeTorus( 2, 1, 32, 32 );
    const Mesh b = makeTorus( 2, 1, 24, 24 );
    FaceBitSet aHalf( a.topology.faceSize() );
    for ( auto f : a.topology.getValidFaces() )
        if ( a.triCenter( f ).x > 0 )
            aHalf.set( f );

    int numColliding = 0;
    for ( float shift : { 0.0f, 0.5f, 1.9f, 3.0f, 5.9f, 6.1f, 10.0f } )
    {
        const auto xf = AffineXf3f::translation( { shift, 0.3f, 0.2f } );
        for ( const FaceBitSet* region : std::array<const FaceBitSet*, 2>{ nullptr, &aHalf } )
        {
            const MeshPart aPart( a, region );
            const auto all = findCollidingTriangles( aPart, b, &xf );
            const auto first = findCollidingTriangles( aPart, b, &xf, true );
            ASSERT_EQ( first.size(), all.empty() ? size_t( 0 ) : size_t( 1 ) );
            if ( first.empty() )
                continue;
            ++numColliding;
            EXPECT_TRUE( std::any_of( all.begin(), all.end(), [&] ( const FaceFace& ff )
                { return ff.aFace == first[0].aFace && ff.bFace == first[0].bFace; } ) );
            const auto again = findCollidingTriangles( aPart, b, &xf, true );
            ASSERT_EQ( again.size(), size_t( 1 ) );
            EXPECT_EQ( again[0].aFace, first[0].aFace );
            EXPECT_EQ( again[0].bFace, first[0].bFace );
        }
    }
    EXPECT_GT( numColliding, 0 );
}

} //namespace MR
