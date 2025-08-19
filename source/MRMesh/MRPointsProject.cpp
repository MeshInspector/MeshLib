#include "MRPointsProject.h"
#include "MRPointCloud.h"
#include "MRAABBTreePoints.h"
#include "MRFewSmallest.h"
#include "MRBuffer.h"
#include "MRBitSetParallelFor.h"
#include "MRInplaceStack.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace {

struct SubTask
{
    NoInitNodeId n;
    float distSq;
};

} //anonymous namespace


PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const PointCloudPart& pcp,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq, VertPredicate skipCb )
{
    const auto& tree = pcp.cloud.getAABBTree();
    return findProjectionOnPoints( pt, tree, upDistLimitSq, xf, loDistLimitSq, pcp.region, skipCb );
}

PointsProjectionResult findProjectionOnPoints( const Vector3f& pt, const AABBTreePoints& tree,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq, const VertBitSet* region, VertPredicate skipCb )
{
    const auto& orderedPoints = tree.orderedPoints();

    PointsProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.nodes().empty() )
        return res;

    InplaceStack<SubTask, 32> subtasks;

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < res.distSq )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
        return SubTask { n, distSq };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
        const auto& node = tree[s.n];
        if ( s.distSq >= res.distSq )
            continue;

        if ( node.leaf() )
        {
            auto [first, last] = node.getLeafPointRange();
            bool lowBreak = false;
            for ( int i = first; i < last && !lowBreak; ++i )
            {
                if ( region && !region->test( orderedPoints[i].id ) )
                    continue;
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

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
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

    InplaceStack<SubTask, 32> subtasks;

    auto topDistSq = [&]
    {
        return !res.full() ? upDistLimitSq : res.top().distSq;
    };

    auto addSubTask = [&] ( const SubTask& s )
    {
        if ( s.distSq < topDistSq() )
            subtasks.push( s );
    };

    auto getSubTask = [&] ( NodeId n )
    {
        const auto & box = tree.nodes()[n].box;
        float distSq = xf ? transformed( box, *xf ).getDistanceSq( pt ) : box.getDistanceSq( pt );
        return SubTask { n, distSq };
    };

    addSubTask( getSubTask( tree.rootNodeId() ) );

    while ( !subtasks.empty() )
    {
        const auto s = subtasks.top();
        subtasks.pop();
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

        auto s1 = getSubTask( node.l );
        auto s2 = getSubTask( node.r );
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
    MR_TIMER;
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
    MR_TIMER;
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

Expected<void> PointsProjector::setPointCloud( const PointCloud& pointCloud )
{
    pointCloud_ = &pointCloud;
    return {};
}

Expected<void> PointsProjector::findProjections( std::vector<PointsProjectionResult>& results,
    const std::vector<Vector3f>& points, const FindProjectionOnPointsSettings& settings ) const
{
    if ( !pointCloud_ )
        return unexpected( "No reference point cloud is set" );

    /// force compute AABB tree
    (void)pointCloud_->getAABBTree();

    results.resize( points.size() );
    ParallelFor( points, [&] ( size_t i )
    {
        if ( settings.valid && !settings.valid->test( i ) )
            return;

        results[i] = findProjectionOnPoints(
            settings.xf ? ( *settings.xf )( points[i] ) : points[i],
            *pointCloud_,
            settings.upDistLimitSq,
            nullptr,
            settings.loDistLimitSq,
            settings.skipSameIndex ? [i] ( VertId v ) { return v == i; } : VertPredicate{}
        );
    },
    settings.cb );

    return {};
}

size_t PointsProjector::projectionsHeapBytes( size_t ) const
{
    return 0;
}

} //namespace MR
