#include "MRFastWindingNumber.h"
#include "MRMesh.h"
#include "MRTimer.h"
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

float FastWindingNumber::Dipole::w( const Vector3f & q ) const
{
    const auto dp = pos() - q;
    const auto d = dp.length();
    return d > 0 ? dot( dp, dirArea ) / ( 4 * PI_F * d * d * d ) : 0;
}

float FastWindingNumber::Dipole::wSubdiv( const Vector3f & q, float beta, const ThreePoints & tri ) const
{
    constexpr int MaxSubdivLevel = 3;
    constexpr int MaxStackElm = 4 * 4 * 4;
    struct DipoleTri
    {
        Dipole d;
        ThreePoints tri;
        int level = 0;
    };
    DipoleTri subtris[MaxStackElm];
    int stackSize = 0;
    subtris[stackSize++] = DipoleTri{ .d = *this, .tri = tri, .level = 0 };
    float res = 0;
    while( stackSize > 0 )
    {
        const auto t = subtris[--stackSize];
        if ( t.level >= MaxSubdivLevel || t.d.goodApprox( q, beta ) )
        {
            res += t.d.w( q );
            continue;
        }
        const auto mid01 = 0.5f * ( t.tri[0] + t.tri[1] );
        const auto mid12 = 0.5f * ( t.tri[1] + t.tri[2] );
        const auto mid20 = 0.5f * ( t.tri[2] + t.tri[0] );
        Dipole sd{ .area = 0.25f * t.d.area, .dirArea = 0.25f * t.d.dirArea, .rr = 0.25f * t.d.rr };
        assert( stackSize + 4 <= MaxStackElm );
        sd.areaPos = sd.area / 3 * ( t.tri[0] + mid01 + mid20 );
        subtris[stackSize++] = DipoleTri{ .d = sd, .tri = { t.tri[0], mid01, mid20 }, .level = t.level + 1 };
        sd.areaPos = sd.area / 3 * ( mid01 + mid12 + mid20 );
        subtris[stackSize++] = DipoleTri{ .d = sd, .tri = { mid01, mid12, mid20 }, .level = t.level + 1 };
        sd.areaPos = sd.area / 3 * ( mid01 + t.tri[1] + mid12 );
        subtris[stackSize++] = DipoleTri{ .d = sd, .tri = { mid01, t.tri[1], mid12 }, .level = t.level + 1 };
        sd.areaPos = sd.area / 3 * ( mid12 + t.tri[2] + mid20 );
        subtris[stackSize++] = DipoleTri{ .d = sd, .tri = { mid12, t.tri[2], mid20 }, .level = t.level + 1 };
    }
    return res;
}

float FastWindingNumber::calc( const Vector3f & q, float beta ) const
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
        res += d.wSubdiv( q, beta, mesh_.getTriPoints( node.leafId() ) );
    }
    return res;
}

} // namespace MR
