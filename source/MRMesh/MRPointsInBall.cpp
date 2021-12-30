#include "MRPointsInBall.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"

namespace MR
{

void findPointsInBall( const PointCloud& pointCloud, const Vector3f& center, float radius, const FoundPointCallback& foundCallback )
{
    if ( !foundCallback )
    {
        assert( false );
        return;
    }

    const AABBTreePoints& tree = pointCloud.getAABBTree();
    if ( tree.nodes().empty() )
    {
        assert( false );
        return;
    }

    const auto& orderedPoints = tree.orderedPoints();
    const float radiusSq = sqr( radius );

    struct SubTask
    {
        AABBTreePoints::NodeId n;
        float distSq = 0;
        SubTask() = default;
        SubTask( AABBTreePoints::NodeId n, float dd ) : n( n ), distSq( dd ){}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&]( const SubTask& s )
    {
        if ( s.distSq < radiusSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&]( AABBTreePoints::NodeId n )
    {
        float distSq = ( tree.nodes()[n].box.getBoxClosestPointTo( center ) - center ).lengthSq();
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = tree[s.n];

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            for ( int i = first; i < last; ++i )
                if ( ( orderedPoints[i].coord - center ).lengthSq() <= radiusSq )
                    foundCallback( orderedPoints[i].id, orderedPoints[i].coord );
            continue;
        }

        auto s1 = getSubTask( node.leftOrFirst );
        auto s2 = getSubTask( node.rightOrLast );
        if ( s1.distSq < s2.distSq )
            std::swap( s1, s2 );
        assert( s1.distSq >= s2.distSq );
        addSubTask( s1 ); // larger distance to look later
        addSubTask( s2 ); // smaller distance to look first
    }
}

}
