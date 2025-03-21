#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"

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

    if ( tree.nodes().empty() )
        return;

    const auto& orderedPoints = tree.orderedPoints();

    constexpr int MaxStackSize = 32; // to avoid allocations
    NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto boxDistSq = [&]( NodeId n ) // squared distance from ball center to the transformed box with interior
    {
        const auto & box = tree.nodes()[n].box;
        return xf ? transformed( box, *xf ).getDistanceSq( ball.center ) : box.getDistanceSq( ball.center );
    };

    auto addSubTask = [&]( NodeId n )
    {
        assert( stackSize < MaxStackSize );
        subtasks[stackSize++] = n;
    };

    addSubTask( tree.rootNodeId() );

    while ( stackSize > 0 )
    {
        const auto n = subtasks[--stackSize];
        const auto& node = tree[n];
        if ( !( boxDistSq( n ) <= ball.radiusSq ) )
            continue;

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            for ( int i = first; i < last; ++i )
            {
                auto coord = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;

                const PointsProjectionResult candidate
                {
                    .distSq = distanceSq( coord, ball.center ),
                    .vId = orderedPoints[i].id
                };
                if ( candidate.distSq <= ball.radiusSq )
                {
                    if ( foundCallback( candidate, coord, ball ) == Processing::Stop )
                        return;
                }
            }
            continue;
        }

        auto lDistSq = boxDistSq( node.leftOrFirst );
        auto rDistSq = boxDistSq( node.rightOrLast );
        /// first go in the node located closer to ball's center (in case the ball will shrink and the other node will be away)
        if ( lDistSq <= rDistSq )
        {
            addSubTask( node.rightOrLast );
            addSubTask( node.leftOrFirst );
        }
        else
        {
            addSubTask( node.leftOrFirst );
            addSubTask( node.rightOrLast );
        }
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
