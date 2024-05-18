#include "MRDipole.h"
#include "MRAABBTree.h"
#include "MRMesh.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

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

} //namespace MR
