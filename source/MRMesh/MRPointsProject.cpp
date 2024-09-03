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
    NodeId n;
    float distSq;
    SubTask() : n( noInit ) {}
    SubTask( NodeId n, float dd ) : n( n ), distSq( dd ) {}
};

} //anonymous namespace

PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloud& pc,
    float upDistLimitSq /*= FLT_MAX*/, 
    const AffineXf3f* xf /*= nullptr*/, 
    float loDistLimitSq /*= 0*/,
    VertPredicate skipCb /*= {}*/ )
{
    const auto& tree = pc.getAABBTree();
    const auto& orderedPoints = tree.orderedPoints();

    PointsProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
        return res;

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

    auto getSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
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
                if ( skipCb && skipCb( orderedPoints[i].id ) )
                    continue;
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
        // add task with smaller distance last to descend there first
        if ( s1.distSq < s2.distSq )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
    }

    return res;
}

void findFewClosestPoints( const Vector3f& pt, const PointCloud& pc, FewSmallest<PointsProjectionResult> & res,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq )
{
    const auto& tree = pc.getAABBTree();
    // must come after getAABBTree() to avoid situations when the same thread executing getAABBTree enters recursively
    // in this function and appends to not-empty res
    res.clear();

    const auto& orderedPoints = tree.orderedPoints();

    if ( tree.nodes().empty() )
        return;

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

    auto getSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
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
        // add task with smaller distance last to descend there first
        if ( s1.distSq < s2.distSq )
        {
            addSubTask( s2 );
            addSubTask( s1 );
        }
        else
        {
            addSubTask( s1 );
            addSubTask( s2 );
        }
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

VertPair findTwoClosestPoints( const PointCloud& pc, const ProgressCallback & progress )
{
    MR_TIMER
    std::atomic<float> minDistSq{ FLT_MAX };
    tbb::enumerable_thread_specific<VertPair> threadData;
    BitSetParallelFor( pc.validPoints, [&]( VertId v )
    {
        float knownDistSq = minDistSq.load( std::memory_order_relaxed );
        auto proj = findProjectionOnPoints( pc.points[v], pc, knownDistSq, nullptr, 0, [v]( VertId x ) { return v == x; } );
        if ( proj.distSq >= knownDistSq )
            return;
        threadData.local() = { v, proj.vId };
        while ( knownDistSq > proj.distSq && !minDistSq.compare_exchange_strong( knownDistSq, proj.distSq ) ) { }
    }, progress );

    float resMinDistSq{ FLT_MAX };
    VertPair res;
    for ( const auto & p : threadData )
    {
        if ( !p.first || !p.second )
            continue;
        float distSq = ( pc.points[p.first] - pc.points[p.second] ).lengthSq();
        if ( distSq < resMinDistSq )
        {
            resMinDistSq = distSq;
            res = p;
        }
    }
    if ( res.second < res.first ) // if not sort we will get dependency on work distribution among threads
        std::swap( res.first, res.second );
    return res;
}

} //namespace MR
