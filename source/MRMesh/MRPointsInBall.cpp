#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"

namespace MR
{

void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius, const FoundPointCallback& foundCallback )
{
    findPointsInBall( pointCloud.getAABBTree(), center, radius, foundCallback );
}

void findPointsInBall( const AABBTreePoints& tree, const Vector3f& center, float radius, const FoundPointCallback& foundCallback )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    if ( tree.nodes().empty() )
    {
        assert( false );
        return;
    }

    const auto& orderedPoints = tree.orderedPoints();
    const float radiusSq = sqr( radius );

    constexpr int MaxStackSize = 32; // to avoid allocations
    AABBTreePoints::NodeId subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( AABBTreePoints::NodeId n )
    {
        float distSq = ( tree.nodes()[n].box.getBoxClosestPointTo( center ) - center ).lengthSq();
        if ( distSq <= radiusSq )
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
                if ( ( orderedPoints[i].coord - center ).lengthSq() <= radiusSq )
                    foundCallback( orderedPoints[i].id, orderedPoints[i].coord );
            continue;
        }

        addSubTask( node.rightOrLast ); // look at right node later
        addSubTask( node.leftOrFirst ); // look at left node first
    }
}

} //namespace MR
