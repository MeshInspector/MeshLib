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

void calcDipoles( Dipoles& dipoles, const AABBTree& tree, const Mesh& mesh )
{
    MR_TIMER;
    dipoles.resize( tree.numNodes() );
    if ( tree.empty() )
        return;
    const auto & simdNodes = tree.simdNodes();
    const auto & leafTris = tree.leafTris();
    const int numInternal = (int)simdNodes.size();

    // the leaves from their triangles
    ParallelFor( size_t( 0 ), tree.simdLeaves().size(), [&]( size_t l )
    {
        Dipole d;
        for ( int t = int( l ) * AABBTree::MaxLeafTris, end = t + AABBTree::MaxLeafTris; t < end && leafTris[t].face; ++t )
        {
            const auto & lt = leafTris[t];
            const auto & a = mesh.points[lt.verts[0]];
            const auto & b = mesh.points[lt.verts[1]];
            const auto & c = mesh.points[lt.verts[2]];
            const auto da = 0.5f * cross( b - a, c - a );
            const auto area = da.length();
            d.pos += area * ( a + b + c ) / 3.0f; // ( area * pos ) for now
            d.area += area;
            d.dirArea += da;
        }
        dipoles[NodeId( numInternal + int( l ) )] = d;
    } );

    // the internal nodes from the last: children follow their parent
    for ( int n = numInternal - 1; n >= 0; --n )
    {
        const auto & sn = simdNodes[n];
        Dipole d;
        for ( int c = 0; c < AABBTree::MaxChildren && sn.child[c]; ++c )
        {
            const auto & dc = dipoles[tree.nodeId( sn.child[c] )];
            d.pos += dc.pos;
            d.area += dc.area;
            d.dirArea += dc.dirArea;
        }
        dipoles[NodeId( n )] = d;
    }

    // center of mass and the distance to the farthest corner for all nodes
    const auto boxes = tree.getNodeBoxes();
    ParallelFor( dipoles, [&]( NodeId i )
    {
        auto & d = dipoles[i];
        if ( d.area > 0 )
            d.pos /= d.area;
        d.rr = distToFarthestCornerSq( boxes[i], d.pos );
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
    if ( dipoles.empty() )
    {
        assert( false );
        return 0;
    }
    assert( dipoles.size() == tree.numNodes() );

    const float betaSq = sqr( beta );
    const auto & simdNodes = tree.simdNodes();
    const auto & leafTris = tree.leafTris();
    // every level adds at most MaxChildren - 1 nodes, and the tree has at most MaxDepth levels
    // only the nodes with not good enough approximation by their dipoles get in the stack
    InplaceStack<int, 64> subtasks;

    float res = 0;
    if ( !dipoles[NodeId( 0 )].addIfGoodApprox( q, betaSq, res ) )
        subtasks.push( tree.rootRef() );
    while ( !subtasks.empty() )
    {
        const int r = subtasks.top();
        subtasks.pop();
        if ( r >= 0 )
        {
            const auto & sn = simdNodes[r];
            for ( int c = AABBTree::MaxChildren - 1; c >= 0; --c )
                if ( sn.child[c] && !dipoles[tree.nodeId( sn.child[c] )].addIfGoodApprox( q, betaSq, res ) )
                    subtasks.push( sn.child[c] );
            continue;
        }
        for ( int t = ~r * AABBTree::MaxLeafTris, end = t + AABBTree::MaxLeafTris; t < end && leafTris[t].face; ++t )
        {
            const auto & lt = leafTris[t];
            if ( lt.face != skipFace )
                res += triangleSolidAngle( q, { mesh.points[lt.verts[0]], mesh.points[lt.verts[1]], mesh.points[lt.verts[2]] } );
        }
    }
    constexpr float INV_4PI = 1.0f / ( 4 * PI_F );
    return INV_4PI * res;
}

} //namespace MR
