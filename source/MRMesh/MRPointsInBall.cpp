#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePointsTraversal.hpp"
#include "MRInplaceStack.h"
#include <limits>

namespace MR
{

static auto newCallback( const FoundPointCallback& foundCallback )
{
    return [&foundCallback]( const PointsProjectionResult & found, const Vector3f& foundXfPos, Ball3f & )
    {
        foundCallback( found.vId, foundXfPos );
        return Processing::Continue;
    };
}

MRMESH_API void findPointsInBall( const PointCloud& pointCloud, const Ball3f & ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( pointCloud.getAABBTree(), ball, foundCallback, xf );
}

void findPointsInBall( const PointCloud& pointCloud, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( pointCloud.getAABBTree(), ball, newCallback( foundCallback ), xf );
}

void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( mesh.getAABBTreePoints(), ball, foundCallback, xf );
}

void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( mesh.getAABBTreePoints(), ball, newCallback( foundCallback ), xf );
}

void findPointsInBall( const AABBTreePoints& tree, Ball3f ball,
    const OnPointInBallFound& foundCallback, const AffineXf3f* xf )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    if ( tree.empty() )
        return;

    using namespace Simd4;
    const auto& orderedPoints = tree.orderedPoints();
    const auto& simdNodes = tree.simdNodes();

    // a node is pushed if its box is not farther than the ball, and checked again when visited only if the ball has changed since then
    struct SubTask
    {
        int ref;
        int parent; // the index of its parent, or -1 for the root
        int slot; // its index in the children of the parent
        int ballVersion;
    };
    // every level adds at most MaxChildren - 1 tasks
    InplaceStack<SubTask, 3 * AABBTreePoints::MaxDepth + 1> subtasks;
    int ballVersion = 0;

    auto boxDistSq = [&]( const Box3f & box ) // squared distance from ball center to the transformed box with interior
    {
        return xf ? transformed( box, *xf ).getDistanceSq( ball.center ) : box.getDistanceSq( ball.center );
    };

    if ( boxDistSq( tree.getBoundingBox() ) <= ball.radiusSq )
        subtasks.push( { tree.rootRef(), -1, 0, ballVersion } );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        if ( s.ballVersion != ballVersion &&
            !( boxDistSq( s.parent < 0 ? tree.getBoundingBox() : simdNodes[s.parent].get( s.slot ) ) <= ball.radiusSq ) )
            continue;

        if ( s.ref < 0 )
        {
            auto [first, last] = tree.leafPointRange( ~s.ref );
            int i = first;
            while ( i < last )
            {
                if ( !xf && i + 4 <= last )
                {
                    // the distances to four points at once, while the ball stays the same
                    const Ball3f ball0 = ball;
                    alignas( 16 ) float d[4];
                    const F dv = pointsDistSq( orderedPoints.data() + i, set1( ball0.center.x ), set1( ball0.center.y ), set1( ball0.center.z ) );
                    const int m = bits( le( dv, set1( ball0.radiusSq ) ) );
                    store( d, dv );
                    int j = 0;
                    while ( j < 4 )
                    {
                        const int k = j++;
                        if ( !( m & ( 1 << k ) ) )
                            continue;
                        const PointsProjectionResult candidate{ .distSq = d[k], .vId = orderedPoints[i + k].id };
                        if ( foundCallback( candidate, orderedPoints[i + k].coord, ball ) == Processing::Stop )
                            return;
                        if ( ball.center != ball0.center || ball.radiusSq != ball0.radiusSq )
                        {
                            ++ballVersion;
                            break;
                        }
                    }
                    i += j;
                    continue;
                }
                auto coord = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;

                const PointsProjectionResult candidate
                {
                    .distSq = distanceSq( coord, ball.center ),
                    .vId = orderedPoints[i].id
                };
                if ( candidate.distSq <= ball.radiusSq )
                {
                    const Ball3f ball0 = ball;
                    if ( foundCallback( candidate, coord, ball ) == Processing::Stop )
                        return;
                    if ( ball.center != ball0.center || ball.radiusSq != ball0.radiusSq )
                        ++ballVersion;
                }
                ++i;
            }
            continue;
        }

        /// first go in the node located closer to ball's center (in case the ball will shrink and the other node will be away)
        const auto & node = simdNodes[s.ref];
        alignas( 16 ) float d[4];
        if ( xf )
        {
            for ( int i = 0; i < AABBTreePoints::MaxChildren; ++i )
                d[i] = node.child[i] ? boxDistSq( node.get( i ) ) : std::numeric_limits<float>::infinity();
        }
        else
            store( d, Simd4::boxDistSq( node.box, set1( ball.center.x ), set1( ball.center.y ), set1( ball.center.z ) ) );
        // the children in the ball in the order of increasing distance (the first child first for equal ones), pushed from the farthest
        int order[4];
        int numCh = 0;
        for ( int i = 0; i < AABBTreePoints::MaxChildren && node.child[i]; ++i )
        {
            if ( !( d[i] <= ball.radiusSq ) )
                continue;
            int k = numCh++;
            for ( ; k > 0 && d[order[k - 1]] > d[i]; --k )
                order[k] = order[k - 1];
            order[k] = i;
        }
        for ( int k = numCh - 1; k >= 0; --k )
            subtasks.push( { node.child[order[k]], s.ref, order[k], ballVersion } );
    }
}

void findPointsInBall( const AABBTreePoints& tree, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }
    findPointsInBall( tree, Ball3f{ ball }, newCallback( foundCallback ), xf );
}

} //namespace MR
