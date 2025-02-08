#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRMesh.h"
#include "MRAABBTreePoints.h"

namespace MR
{

void findPointsInBall( const PointCloud& pointCloud, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( pointCloud.getAABBTree(), ball, foundCallback, xf );
}

void findPointsInBall( const Mesh& mesh, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
{
    findPointsInBall( mesh.getAABBTreePoints(), ball, foundCallback, xf );
}

void findPointsInBall( const AABBTreePoints& tree, const Ball3f& ball,
    const FoundPointCallback& foundCallback, const AffineXf3f* xf )
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

    auto addSubTask = [&]( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( ball.center ) : box.getDistanceSq( ball.center );
        if ( distSq <= ball.radiusSq )
            subtasks[stackSize++] = n;
    };

    addSubTask( tree.rootNodeId() );

    while ( stackSize > 0 )
    {
        const auto n = subtasks[--stackSize];
        const auto& node = tree[n];

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            for ( int i = first; i < last; ++i )
            {
                auto coord = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
                if ( !ball.outside( coord ) )
                    foundCallback( orderedPoints[i].id, coord );
            }
            continue;
        }

        addSubTask( node.rightOrLast ); // look at right node later
        addSubTask( node.leftOrFirst ); // look at left node first
    }
}

} //namespace MR
