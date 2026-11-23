#pragma once

#include "MRAABBTreePoints.h"
#include "MRSimd4.hpp"

namespace MR
{

/// the squared distances from (px, py, pz) to the four points starting from p, as ( p[i].coord - pt ).lengthSq() but maybe without FMA,
/// so they only select the candidates, multiplied by cSimdDistLow, and the exact distances are computed again by the scalar formula
inline Simd4::F pointsDistSq( const AABBTreePoints::Point * p, Simd4::F px, Simd4::F py, Simd4::F pz )
{
    using namespace Simd4;
    static_assert( sizeof( AABBTreePoints::Point ) == 4 * sizeof( float ) );
    F r[4] = { loadu( &p[0].coord.x ), loadu( &p[1].coord.x ), loadu( &p[2].coord.x ), loadu( &p[3].coord.x ) };
    transpose( r );
    const F dx = sub( r[0], px ), dy = sub( r[1], py ), dz = sub( r[2], pz );
    return add( add( mul( dx, dx ), mul( dy, dy ) ), mul( dz, dz ) );
}

/// the factor of SIMD distances (or of the bound of their rounding) to never miss a point that the scalar formula finds closer
constexpr float cSimdDistLow = 1 - 1e-6f;

/// a node of AABBTreePoints in a traversal: its reference as in AABBTreePoints::SimdNode::child and the value deciding its order
struct PointsSubTask
{
    int ref;
    float distSq;
};

/// puts in ch the children of node with d[i] < limit in the order of decreasing d[i] (stable for equal values), and returns their number;
/// a traversal pushes all but the last in its stack and descends in the last one
inline int sortedChildren( const AABBTreePoints::SimdNode & node, const float ( &d )[4], float limit, PointsSubTask ( &ch )[4] )
{
    int numCh = 0;
    for ( int i = 0; i < AABBTreePoints::MaxChildren && node.child[i]; ++i )
    {
        if ( !( d[i] < limit ) )
            continue;
        int k = numCh++;
        for ( ; k > 0 && ch[k - 1].distSq < d[i]; --k )
            ch[k] = ch[k - 1];
        ch[k] = { node.child[i], d[i] };
    }
    return numCh;
}

} // namespace MR
