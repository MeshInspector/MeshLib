#include "MRDipole.h"
#include "MRAABBTree.h"
#include "MRAABBTreeWide.h"
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

void calcDipoles( Dipoles& dipoles, const AABBTreeWide& tree, const Mesh& mesh )
{
    MR_TIMER;
    const auto & nodes = tree.nodes();
    dipoles.resize( nodes.size() );

    // the leaves from their triangles
    ParallelFor( dipoles, [&]( NodeId i )
    {
        const auto & node = nodes[i];
        if ( !node.leaf() )
            return;
        Dipole d;
        for ( int t = node.first, end = node.first + node.numLeafTris(); t < end; ++t )
        {
            const auto & lt = tree.leafTris()[t];
            const auto & a = mesh.points[lt.verts[0]];
            const auto & b = mesh.points[lt.verts[1]];
            const auto & c = mesh.points[lt.verts[2]];
            const auto da = 0.5f * cross( b - a, c - a );
            const auto area = da.length();
            d.pos += area * ( a + b + c ) / 3.0f; // ( area * pos ) for now
            d.area += area;
            d.dirArea += da;
        }
        dipoles[i] = d;
    } );

    // the other nodes from the last: children follow their parent
    for ( auto i = dipoles.backId(); i.valid(); --i )
    {
        const auto & node = nodes[i];
        if ( node.leaf() )
            continue;
        Dipole d;
        for ( int c = 0; c < node.count; ++c )
        {
            const auto & dc = dipoles[NodeId( node.first + c )];
            d.pos += dc.pos;
            d.area += dc.area;
            d.dirArea += dc.dirArea;
        }
        dipoles[i] = d;
    }

    // center of mass and the distance to the farthest corner for all nodes
    ParallelFor( dipoles, [&]( NodeId i )
    {
        auto & d = dipoles[i];
        if ( d.area > 0 )
            d.pos /= d.area;
        d.rr = distToFarthestCornerSq( nodes[i].box, d.pos );
    } );
}

Dipoles calcDipoles( const AABBTreeWide& tree, const Mesh& mesh )
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

float calcFastWindingNumber( const Dipoles& dipoles, const AABBTreeWide& tree, const Mesh& mesh,
    const Vector3f & q, float beta, FaceId skipFace )
{
    if ( dipoles.empty() )
    {
        assert( false );
        return 0;
    }

    const float betaSq = sqr( beta );
    const auto & nodes = tree.nodes();
    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    InplaceStack<NoInitNodeId, 64> subtasks;
    subtasks.push( NodeId( 0 ) );

    float res = 0;
    while ( !subtasks.empty() )
    {
        const NodeId i = subtasks.top();
        subtasks.pop();
        if ( dipoles[i].addIfGoodApprox( q, betaSq, res ) )
            continue;
        const auto & node = nodes[i];
        if ( !node.leaf() )
        {
            for ( int c = node.count - 1; c >= 0; --c )
                subtasks.push( NodeId( node.first + c ) );
            continue;
        }
        for ( int t = node.first, end = node.first + node.numLeafTris(); t < end; ++t )
        {
            const auto & lt = tree.leafTris()[t];
            if ( lt.face != skipFace )
                res += triangleSolidAngle( q, { mesh.points[lt.verts[0]], mesh.points[lt.verts[1]], mesh.points[lt.verts[2]] } );
        }
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    return INV_4PI * res;
}

} //namespace MR
