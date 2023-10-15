#include "MRPointsProject.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRFewSmallest.h"
#include "MRBuffer.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace {

struct SubTask
{
    AABBTreePoints::NodeId n;
    float distSq = 0;
    SubTask() = default;
    SubTask( AABBTreePoints::NodeId n, float dd ) : n( n ), distSq( dd )
    {}
};

} //anonymous namespace

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
                    if ( lowBreak )
                        break;
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

void findFewClosestPoints( const Vector3f& pt, const PointCloud& pc, FewSmallest<PointsProjectionResult> & res,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq )
{
    const auto& tree = pc.getAABBTree();
    const auto& orderedPoints = tree.orderedPoints();

    if ( tree.nodes().empty() )
    {
        assert( false );
        return;
    }

    constexpr int MaxStackSize = 32; // to avoid allocations
    SubTask subtasks[MaxStackSize];
    int stackSize = 0;

    auto topDistSq = [&]
    {
        return !res.full() ? upDistLimitSq : res.top().distSq;
    };

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < topDistSq() )
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
        if ( s.distSq >= topDistSq() )
            continue;

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            bool lowBreak = false;
            for ( int i = first; i < last && !lowBreak; ++i )
            {
                auto proj = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
                float distSq = ( proj - pt ).lengthSq();
                if ( distSq < topDistSq() )
                {
                    res.push( { .distSq = distSq, .vId = orderedPoints[i].id } );
                    lowBreak = res.full() && res.top().distSq <= loDistLimitSq;
                    if ( lowBreak )
                        break;
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
}

Buffer<VertId> findNClosestPointsPerPoint( const PointCloud& pc, int numNei, const ProgressCallback & progress )
{
    MR_TIMER
    assert( numNei >= 1 );
    Buffer<VertId> res( pc.points.size() * numNei );

    tbb::enumerable_thread_specific<FewSmallest<PointsProjectionResult>> perThreadNeis( numNei + 1 );

    pc.getAABBTree(); // to avoid multiple calls to tree construction from parallel region,
                      // which can result that two different vertices will start being processed by one thread

    if ( !BitSetParallelFor( pc.validPoints, [&]( VertId v )
    {
        auto & neis = perThreadNeis.local();
        neis.clear();
        assert( neis.maxElms() == numNei + 1 );
        findFewClosestPoints( pc.points[v], pc, neis );
        VertId * p = res.data() + ( (size_t)v * numNei );
        const VertId * pEnd = p + numNei;
        for ( const auto & n : neis.get() )
            if ( n.vId != v && p < pEnd )
                *p++ = n.vId;
        while ( p < pEnd )
            *p++ = {};
    }, progress ) )
        res.clear();

    return res;
}

} //namespace MR
