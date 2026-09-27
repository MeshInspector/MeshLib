#include <MRMesh/MRChamferEdges.h>
#include <MRMesh/MRMesh.h>
#include <MRMesh/MRCube.h>
#include <MRMesh/MRBitSet.h>
#include <MRMesh/MRVector2.h>
#include <MRMesh/MREdgeIterator.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>

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
        for ( auto f : mesh.topology.getValidFaces() )
        {
            Vector3f a, b, c;
            mesh.getTriPoints( f, a, b, c );
            EXPECT_NEAR( chamferedCubeFunc( ( a + b + c ) / 3.0f, d ), 0, 1e-5f );
        }
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
