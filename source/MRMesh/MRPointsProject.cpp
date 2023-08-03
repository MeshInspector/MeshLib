#include "MRPointsProject.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"

namespace MR
{

PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloud& pc,
    float upDistLimitSq /*= FLT_MAX*/, 
    const AffineXf3f* xf /*= nullptr*/, 
    float loDistLimitSq /*= 0 */ )
{
    const auto& tree = pc.getAABBTree();
    const auto& orderedPoints = tree.orderedPoints();

    PointsProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
    {
        assert( false );
        return res;
    }

    struct SubTask
    {
        AABBTreePoints::NodeId n;
        float distSq = 0;
        SubTask() = default;
        SubTask( AABBTreePoints::NodeId n, float dd ) : n( n ), distSq( dd )
        {}
    };

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
        {
            assert( stackSize < MaxStackSize );
            subtasks[stackSize++] = s;
        }
    };

    auto getSubTask = [&] ( AABBTreePoints::NodeId n )
    {
        float distSq = ( transformed( tree.nodes()[n].box, xf ).getBoxClosestPointTo( pt ) - pt ).lengthSq();
        return SubTask( n, distSq );
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( stackSize > 0 )
    {
        const auto s = subtasks[--stackSize];
        const auto& node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            bool lowBreak = false;
            for ( int i = first; i < last && !lowBreak; ++i )
            {
                auto proj = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
                float distSq = ( proj - pt ).lengthSq();
                if ( distSq < res.distSq )
                {
                    res.distSq = distSq;
                    res.vId = orderedPoints[i].id;
                    lowBreak = distSq <= loDistLimitSq;
                }
            }
            if ( lowBreak )
                break;
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

    return res;
}

}