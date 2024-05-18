#include "MRDipole.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRGTest.h"

namespace MR
{

static float distToFarthestCornerSq( const Box3f & box, const Vector3f & pos )
{
    float res = 0;
    for ( int i = 0; i < 3; ++i )
    {
        const auto toMinSq = sqr( pos[i] - box.min[i] );
        const auto toMaxSq = sqr( pos[i] - box.max[i] );
        res += std::max( toMinSq, toMaxSq );
    }
    return res;
}

void calcDipoles( Dipoles& dipoles, const AABBTree& tree_, const Mesh& mesh )
{
    MR_TIMER
    dipoles.resize( tree_.nodes().size() );

    // compute dipole data for tree leaves
    ParallelFor( dipoles, [&]( NodeId i )
    {
        const auto& node = tree_[i];
        if ( !node.leaf() )
            return;
        const FaceId f = node.leafId();
        const auto da = 0.5f * mesh.dirDblArea( f );
        const auto a = da.length();
        const auto ap = a * mesh.triCenter( f );
        dipoles[i] = Dipole
        {
            .areaPos = ap,
            .area = a,
            .dirArea = da
        };
    } );

    // compute dipole data for not-leaf tree nodes
    for ( auto i = dipoles.backId(); i; --i )
    {
        const auto& node = tree_[i];
        if ( node.leaf() )
            continue;
        const auto& dl = dipoles[node.l];
        const auto& dr = dipoles[node.r];
        dipoles[i] = Dipole
        {
            .areaPos = dl.areaPos + dr.areaPos,
            .area = dl.area + dr.area,
            .dirArea = dl.dirArea + dr.dirArea
        };
    }

    // compute distance to farthest corner for all nodes
    ParallelFor( dipoles, [&]( NodeId i )
    {
        const auto& node = tree_[i];
        auto& d = dipoles[i];
        d.rr = distToFarthestCornerSq( node.box, d.pos() );
    } );
}

constexpr float INV_4PI = 1.0f / ( 4 * PI_F );

float Dipole::w( const Vector3f & q ) const
{
    const auto dp = pos() - q;
    const auto d = dp.length();
    return d > 0 ? INV_4PI * dot( dp, dirArea ) / ( d * d * d ) : 0;
}

/// see (6) in https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.pdf
static float triangleSolidAngle( const Vector3f & p, const Triangle3f & tri )
{
    Matrix3f m;
    m.x = tri[0] - p;
    m.y = tri[1] - p;
    m.z = tri[2] - p;
    auto x = m.x.length();
    auto y = m.y.length();
    auto z = m.z.length();
    auto den = x * y * z + dot( m.x, m.y ) * z + dot( m.y, m.z ) * x + dot( m.z, m.x ) * y;
    return 2 * std::atan2( m.det(), den );
}

float calcFastWindingNumber( const Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh,
    const Vector3f & q, float beta, FaceId skipFace )
{
    float res = 0;
    if ( dipoles.empty() )
    {
        assert( false );
        return res;
    }

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;
    subtasks[stackSize++] = tree.rootNodeId();

    while( stackSize > 0 )
    {
        const auto i = subtasks[--stackSize];
        const auto & node = tree[i];
        const auto & d = dipoles[i];
        if ( d.goodApprox( q, beta ) )
        {
            res += d.w( q );
            continue;
        }
        if ( !node.leaf() )
        {
            // recurse deeper
            subtasks[stackSize++] = node.r; // to look later
            subtasks[stackSize++] = node.l; // to look first
            continue;
        }
        if ( node.leafId() != skipFace )
            res += INV_4PI * triangleSolidAngle( q, mesh.getTriPoints( node.leafId() ) );
    }
    return res;
}

TEST(MRMesh, TriangleSolidAngle) 
{
    const Triangle3f tri =
    {
        Vector3f{ 0.0f, 0.0f, 0.0f },
        Vector3f{ 1.0f, 0.0f, 0.0f },
        Vector3f{ 0.0f, 1.0f, 0.0f }
    };
    const auto c = ( tri[0] + tri[1] + tri[2] ) / 3.0f;

    // solid angle near triangle center abruptly changes from -2pi to 2pi when the point crosses the triangle plane
    const auto x = triangleSolidAngle( c + Vector3f( 0, 0, 1e-5f ), tri );
    EXPECT_NEAR( x, -2 * PI_F, 1e-3f );
    auto y = triangleSolidAngle( c - Vector3f( 0, 0, 1e-5f ), tri );
    EXPECT_NEAR( y,  2 * PI_F, 1e-3f );

    // solid angle in triangle vertices is equal to zero exactly
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( triangleSolidAngle( tri[i], tri ), 0 );
    }

    // solid angle in the triangle plane outside of triangle is equal to zero exactly
    EXPECT_EQ( triangleSolidAngle( tri[1] + tri[2], tri ), 0 );
    EXPECT_EQ( triangleSolidAngle( -tri[1], tri ), 0 );
    EXPECT_EQ( triangleSolidAngle( -tri[2], tri ), 0 );
}

} //namespace MR
