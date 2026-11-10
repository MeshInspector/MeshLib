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

    auto add = []( Dipole & to, const Dipole & d )
    {
        to.pos += d.pos; // ( area * pos ) for now
        to.area += d.area;
        to.dirArea += d.dirArea;
    };

    // add dipole data of leaves to their parent nodes
    ParallelFor( dipoles, [&]( NodeId i )
    {
        const auto& node = tree_[i];
        Dipole d;
        for ( auto c : { node.l, node.r } )
        {
            if ( !AABBTree::isLeaf( c ) )
                continue;
            const auto & v = tree_.leafVerts( c );
            const auto & p0 = mesh.points[v[0]];
            const auto & p1 = mesh.points[v[1]];
            const auto & p2 = mesh.points[v[2]];
            const auto da = 0.5f * cross( p1 - p0, p2 - p0 );
            const auto a = da.length();
            add( d, Dipole{ .pos = a * ( p0 + p1 + p2 ) / 3.0f, .area = a, .dirArea = da } );
        }
        dipoles[i] = d;
    } );

    // add dipole data of not-leaf children nodes
    for ( auto i = NodeId( int( dipoles.size() ) - 1 ); i >= 0; --i )
    {
        const auto& node = tree_[i];
        for ( auto c : { node.l, node.r } )
            if ( !AABBTree::isLeaf( c ) )
                add( dipoles[i], dipoles[c] );
    }

    // compute 1) center mass 2) distance to farthest corner for all nodes
    ParallelFor( dipoles, [&]( NodeId i )
    {
        auto& d = dipoles[i];
        if ( d.area > 0 )
            d.pos /= d.area;
        d.rr = distToFarthestCornerSq( tree_.box( i ), d.pos );
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
    if ( tree.empty() )
        return 0;
    assert( dipoles.size() == tree.nodes().size() );

    const float betaSq = sqr( beta );
    InplaceStack<NoInitNodeId, 32> subtasks;
    subtasks.push( tree.rootNodeId() );

    float res = 0;
    while ( !subtasks.empty() )
    {
        NodeId i = subtasks.top();
        subtasks.pop();
        if ( AABBTree::isLeaf( i ) )
        {
            if ( tree.leafId( i ) != skipFace )
            {
                const auto & v = tree.leafVerts( i );
                res += triangleSolidAngle( q, { mesh.points[v[0]], mesh.points[v[1]], mesh.points[v[2]] } );
            }
            continue;
        }
        if ( dipoles[i].addIfGoodApprox( q, betaSq, res ) )
            continue;
        // recurse deeper
        const auto & node = tree[i];
        subtasks.push( node.r ); // to look later
        subtasks.push( node.l ); // to look first
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    return INV_4PI * res;
}

} //namespace MR
