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
    const auto & nodes = tree.nodes();
    // every level adds at most MaxChildren - 1 nodes, and the tree of 2^31 triangles has 16 levels
    // only the nodes with not good enough approximation by their dipoles get in the stack
    InplaceStack<NoInitNodeId, 64> subtasks;

    float res = 0;
    if ( !dipoles[NodeId( 0 )].addIfGoodApprox( q, betaSq, res ) )
        subtasks.push( NodeId( 0 ) );
    while ( !subtasks.empty() )
    {
        const auto & node = nodes[subtasks.top()];
        subtasks.pop();
        if ( !node.leaf() )
        {
            for ( int c = node.count - 1; c >= 0; --c )
                if ( const NodeId n( node.first + c ); !dipoles[n].addIfGoodApprox( q, betaSq, res ) )
                    subtasks.push( n );
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
