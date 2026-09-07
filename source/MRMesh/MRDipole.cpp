#include "MRDipole.h"
#include "MRAABBTree.h"
#include "MRInplaceStack.h"
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
    MR_TIMER;
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
            .pos = ap, // ( area * pos ) for now
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
            .pos = dl.pos + dr.pos, // ( area * pos ) for now
            .area = dl.area + dr.area,
            .dirArea = dl.dirArea + dr.dirArea
        };
    }

    // compute 1) center mass 2) distance to farthest corner for all nodes
    ParallelFor( dipoles, [&]( NodeId i )
    {
        const auto& node = tree_[i];
        auto& d = dipoles[i];
        if ( d.area > 0 )
            d.pos /= d.area;
        d.rr = distToFarthestCornerSq( node.box, d.pos );
    } );
}

Dipoles calcDipoles( const AABBTree& tree, const Mesh& mesh )
{
    Dipoles dipoles;
    calcDipoles( dipoles, tree, mesh );
    return dipoles;
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
    if ( dipoles.empty() )
    {
        assert( false );
        return 0;
    }

    const float betaSq = sqr( beta );
    InplaceStack<NoInitNodeId, 32> subtasks;
    subtasks.push( AABBTree::rootNodeId() );

    float res = 0;
    while ( !subtasks.empty() )
    {
        NodeId i = subtasks.top();
        subtasks.pop();
        const auto & node = tree[i];
        const auto & d = dipoles[i];
        if ( d.addIfGoodApprox( q, betaSq, res ) )
            continue;
        if ( !node.leaf() )
        {
            // recurse deeper
            subtasks.push( node.r ); // to look later
            subtasks.push( node.l ); // to look first
            continue;
        }
        if ( node.leafId() != skipFace )
            res += triangleSolidAngle( q, mesh.getTriPoints( node.leafId() ) );
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    return INV_4PI * res;
}

} //namespace MR
