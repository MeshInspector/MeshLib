#include "MRFastWindingNumber.h"
#include "MRMesh.h"
#include "MRTimer.h"
#include "MRGTest.h"
#include "MRPch/MRTBB.h"

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

FastWindingNumber::FastWindingNumber( const Mesh & mesh ) : mesh_( mesh ), tree_( mesh.getAABBTree() )
{
    MR_TIMER
    dipoles_.resize( tree_.nodes().size() );

    // compute dipole data for tree leaves
    tbb::parallel_for( tbb::blocked_range<NodeId>( NodeId{ 0 }, dipoles_.endId() ),
        [&]( const tbb::blocked_range<NodeId>& range )
    {
        for ( NodeId i = range.begin(); i < range.end(); ++i )
        {
            const auto & node = tree_[ i ];
            if ( !node.leaf() )
                continue;
            const FaceId f = node.leafId();
            const auto da = 0.5f * mesh.dirDblArea( f );
            const auto a = da.length();
            const auto ap = a * mesh.triCenter( f );
            dipoles_[ i ] = Dipole
            {
                .areaPos = ap,
                .area = a,
                .dirArea = da
            };
        }
    } );

    // compute dipole data for not-leaf tree nodes
    for ( NodeId i = dipoles_.backId(); i; --i )
    {
        const auto & node = tree_[ i ];
        if ( node.leaf() )
            continue;
        const auto & dl = dipoles_[ node.l ];
        const auto & dr = dipoles_[ node.r ];
        dipoles_[ i ] = Dipole
        {
            .areaPos = dl.areaPos + dr.areaPos,
            .area = dl.area + dr.area,
            .dirArea = dl.dirArea + dr.dirArea
        };
    }

    // compute distance to farthest corner for all nodes
    tbb::parallel_for( tbb::blocked_range<NodeId>( NodeId{ 0 }, dipoles_.endId() ),
        [&]( const tbb::blocked_range<NodeId>& range )
    {
        for ( NodeId i = range.begin(); i < range.end(); ++i )
        {
            const auto & node = tree_[ i ];
            auto & d = dipoles_[ i ];
            d.rr = distToFarthestCornerSq( node.box, d.pos() );
        }
    } );
}

constexpr float INV_4PI = 1.0f / ( 4 * PI_F );

float FastWindingNumber::Dipole::w( const Vector3f & q ) const
{
    const auto dp = pos() - q;
    const auto d = dp.length();
    return d > 0 ? INV_4PI * dot( dp, dirArea ) / ( d * d * d ) : 0;
}

/// see (6) in https://users.cs.utah.edu/~ladislav/jacobson13robust/jacobson13robust.pdf
static float triangleSolidAngle( const Vector3f & p, const ThreePoints & tri )
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

float FastWindingNumber::calc( const Vector3f & q, float beta, FaceId skipFace ) const
{
    float res = 0;
    if ( dipoles_.empty() )
    {
        assert( false );
        return res;
    }

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;
    subtasks[stackSize++] = tree_.rootNodeId();

    while( stackSize > 0 )
    {
        const auto i = subtasks[--stackSize];
        const auto & node = tree_[i];
        const auto & d = dipoles_[i];
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
            res += INV_4PI * triangleSolidAngle( q, mesh_.getTriPoints( node.leafId() ) );
    }
    return res;
}

TEST(MRMesh, TriangleSolidAngle) 
{
    const ThreePoints tri =
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

} // namespace MR
