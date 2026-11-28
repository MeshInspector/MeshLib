#include <MRMesh/MRChamferEdges.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MREdgeIterator.h>
#include <MRMesh/MRMeshComponents.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <functional>

namespace MR
{

namespace
{

// sharp edges of the mesh, which have both ends at the given z
UndirectedEdgeBitSet sharpLoop( const Mesh & mesh, float z )
{
    auto res = mesh.findCreaseEdges( 0.1f );
    for ( auto ue : res )
        if ( mesh.orgPnt( ue ).z != z || mesh.destPnt( ue ).z != z )
            res.reset( ue );
    return res;
}

// signed distance-like function of the unit cube with the top loop of edges chamfered at distance d, zero on its surface
float chamferedCubeFunc( const Vector3f & p, float d )
{
    const float s = std::sqrt( 0.5f );
    return std::max( { std::abs( p.x ) - 0.5f, std::abs( p.y ) - 0.5f, p.z - 0.5f, -p.z - 0.5f,
        ( std::abs( p.x ) + p.z - ( 1 - d ) ) * s, ( std::abs( p.y ) + p.z - ( 1 - d ) ) * s } );
}

// volume of the unit cube after chamfering its top loop of edges at distance d
float chamferedCubeVolume( float d )
{
    return 1 - d + d / 3 * ( 1 + sqr( 1 - 2 * d ) + ( 1 - 2 * d ) );
}

// the number of edges, where the triangles on both sides are folded onto one another
int countFolds( const Mesh & mesh )
{
    int res = 0;
    for ( auto ue : undirectedEdges( mesh.topology ) )
        if ( auto l = mesh.topology.left( ue ), r = mesh.topology.right( ue ); l && r && dot( mesh.normal( l ), mesh.normal( r ) ) < -0.5f )
            ++res;
    return res;
}

// cube without the faces with given normal directions
Mesh makeOpenCube( std::function<bool( const Vector3f & )> removeFaceWithNormal )
{
    auto mesh = makeCube();
    FaceBitSet remove( mesh.topology.faceSize() );
    for ( auto f : mesh.topology.getValidFaces() )
        if ( removeFaceWithNormal( mesh.normal( f ) ) )
            remove.set( f );
    mesh.deleteFaces( remove );
    return mesh;
}

// maximal deviation of the triangle centroids from the surface given as the zero level of the function
float maxCentroidDeviation( const Mesh & mesh, std::function<float( const Vector3f & )> func )
{
    float res = 0;
    for ( auto f : mesh.topology.getValidFaces() )
        res = std::max( res, std::abs( func( mesh.triCenter( f ) ) ) );
    return res;
}

} // anonymous namespace

TEST( MRMesh, ChamferEdges )
{
    for ( float d : { 0.1f, 0.25f } )
    {
        auto mesh = makeCube();
        const auto top = sharpLoop( mesh, 0.5f );
        EXPECT_EQ( top.count(), 4 );

        auto res = chamferEdges( mesh, top, d );
        ASSERT_TRUE( res.has_value() );
        EXPECT_TRUE( res->any() );
        EXPECT_NEAR( mesh.volume(), chamferedCubeVolume( d ), 1e-5 );
        EXPECT_EQ( countFolds( mesh ), 0 );

        // not only the vertices, but the whole triangles lie on the chamfered cube, even at the corners
        EXPECT_LT( maxCentroidDeviation( mesh, [d]( const Vector3f & p ) { return chamferedCubeFunc( p, d ); } ), 1e-5f );
    }
}

TEST( MRMesh, ChamferEdgesTwoLoops )
{
    const float d = 0.1f;
    auto mesh = makeCube();
    auto loops = sharpLoop( mesh, 0.5f ) | sharpLoop( mesh, -0.5f );
    EXPECT_EQ( loops.count(), 8 );

    auto res = chamferEdges( mesh, loops, d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_NEAR( mesh.volume(), 2 * chamferedCubeVolume( d ) - 1, 1e-5 );
}

TEST( MRMesh, ChamferEdgesReflexCorner )
{
    // L-shaped prism, its top loop has one reflex corner at (0,0)
    const Vector2f c[6] = { { -0.5f, -0.5f }, { 0.5f, -0.5f }, { 0.5f, 0 }, { 0, 0 }, { 0, 0.5f }, { -0.5f, 0.5f } };
    VertCoords points;
    for ( float z : { -0.5f, 0.5f } )
        for ( auto p : c )
            points.emplace_back( p.x, p.y, z );
    Triangulation t;
    auto tri = [&]( int a, int b, int d ) { t.push_back( { VertId( a ), VertId( b ), VertId( d ) } ); };
    for ( int i : { 4, 5, 0, 1 } )
    {
        const int j = ( i + 1 ) % 6;
        tri( 9, 6 + i, 6 + j );
        tri( 3, j, i );
    }
    for ( int i = 0; i < 6; ++i )
    {
        const int j = ( i + 1 ) % 6;
        tri( i, j, j + 6 );
        tri( i, j + 6, i + 6 );
    }
    auto mesh = Mesh::fromTriangles( std::move( points ), t );
    EXPECT_NEAR( mesh.volume(), 0.75f, 1e-6 );

    const float d = 0.1f;
    auto res = chamferEdges( mesh, sharpLoop( mesh, 0.5f ), d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( countFolds( mesh ), 0 );

    // at the reflex corner the border is an arc, so a little less than the mitered volume P*d^2/2 - 4*d^3/3 is removed
    const float mitered = 4 * sqr( d ) / 2 - 4 * d * sqr( d ) / 3;
    EXPECT_GT( 0.75f - mesh.volume(), 0.98f * mitered );
    EXPECT_LT( 0.75f - mesh.volume(), mitered );
}

TEST( MRMesh, ChamferEdgesOpenChains )
{
    const float d = 0.1f;
    const float s = std::sqrt( 0.5f );

    // square tube along X: two straight chains from one mesh boundary to the other
    auto tube = makeOpenCube( []( const Vector3f & n ) { return std::abs( n.x ) > 0.9f; } );
    auto res = chamferEdges( tube, sharpLoop( tube, 0.5f ), d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( countFolds( tube ), 0 );
    EXPECT_LT( maxCentroidDeviation( tube, [&]( const Vector3f & p )
    {
        return std::max( { std::abs( p.y ) - 0.5f, std::abs( p.z ) - 0.5f, ( std::abs( p.y ) + p.z - ( 1 - d ) ) * s } );
    } ), 1e-5f );

    // box without the front face: one chain of three top edges with two corners
    auto box = makeOpenCube( []( const Vector3f & n ) { return n.y < -0.9f; } );
    const auto chain = sharpLoop( box, 0.5f );
    EXPECT_EQ( chain.count(), 3 );
    res = chamferEdges( box, chain, d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( countFolds( box ), 0 );
    EXPECT_LT( maxCentroidDeviation( box, [&]( const Vector3f & p )
    {
        return std::max( { std::abs( p.x ) - 0.5f, std::abs( p.y ) - 0.5f, std::abs( p.z ) - 0.5f,
            ( std::abs( p.x ) + p.z - ( 1 - d ) ) * s, ( p.y + p.z - ( 1 - d ) ) * s } );
    } ), 1e-5f );
}

TEST( MRMesh, ChamferEdgesCloseLoops )
{
    // a thin plate: its top and bottom loops are closer than 2*d, so both chamfers are narrowed to 0.45 of the gap
    const float h = 0.075f, d = 0.1f, w = 0.45f * 2 * h;
    const float s = std::sqrt( 0.5f );
    auto mesh = makeCube( Vector3f( 1, 1, 2 * h ), Vector3f( -0.5f, -0.5f, -h ) );
    auto res = chamferEdges( mesh, sharpLoop( mesh, h ) | sharpLoop( mesh, -h ), d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( countFolds( mesh ), 0 );
    EXPECT_LT( maxCentroidDeviation( mesh, [&]( const Vector3f & p )
    {
        return std::max( { std::abs( p.x ) - 0.5f, std::abs( p.y ) - 0.5f, std::abs( p.z ) - h,
            ( std::abs( p.x ) + std::abs( p.z ) - ( 0.5f + h - w ) ) * s, ( std::abs( p.y ) + std::abs( p.z ) - ( 0.5f + h - w ) ) * s } );
    } ), 1e-5f );
}

TEST( MRMesh, ChamferEdgesLoopsJustApart )
{
    // the top and bottom loops of the plate are a bit farther than 2*d, the chamfers are narrowed anyway not to touch each other
    const float d = 0.1f, h = 1.02f * d, w = 0.45f * 2 * h;
    const float s = std::sqrt( 0.5f );
    auto mesh = makeCube( Vector3f( 1, 1, 2 * h ), Vector3f( -0.5f, -0.5f, -h ) );
    auto res = chamferEdges( mesh, sharpLoop( mesh, h ) | sharpLoop( mesh, -h ), d );
    ASSERT_TRUE( res.has_value() );
    EXPECT_EQ( MeshComponents::getNumComponents( { mesh, &*res } ), 2 );
    EXPECT_LT( maxCentroidDeviation( mesh, [&]( const Vector3f & p )
    {
        return std::max( { std::abs( p.x ) - 0.5f, std::abs( p.y ) - 0.5f, std::abs( p.z ) - h,
            ( std::abs( p.x ) + std::abs( p.z ) - ( 0.5f + h - w ) ) * s, ( std::abs( p.y ) + std::abs( p.z ) - ( 0.5f + h - w ) ) * s } );
    } ), 1e-5f );
}

TEST( MRMesh, ChamferEdgesDoesNotFit )
{
    // the side faces of a thin plate are lower than the distance, so the chamfer of its top loop would reach the bottom face
    const float h = 0.05f;
    auto mesh = makeCube( Vector3f( 1, 1, 2 * h ), Vector3f( -0.5f, -0.5f, -h ) );
    auto res = chamferEdges( mesh, sharpLoop( mesh, h ), 0.2f );
    ASSERT_FALSE( res.has_value() );
    EXPECT_NE( res.error().find( "does not fit" ), std::string::npos );
}

TEST( MRMesh, ChamferEdgesBadInput )
{
    auto mesh = makeCube();
    auto top = sharpLoop( mesh, 0.5f );
    EXPECT_FALSE( chamferEdges( mesh, top, 0 ).has_value() );

    top.reset( top.find_first() );
    EXPECT_FALSE( chamferEdges( mesh, top, 0.1f ).has_value() );
    EXPECT_FALSE( chamferEdges( mesh, mesh.findCreaseEdges( 0.1f ), 0.1f ).has_value() );
}

} // namespace MR
