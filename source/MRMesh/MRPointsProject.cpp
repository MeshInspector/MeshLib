#include "MRPointsProject.h"
#include "MRPointCloud.h"
#include "MRAABBTreePointsTraversal.hpp"
#include "MRFewSmallest.h"
#include "MRBuffer.h"
#include "MRBitSetParallelFor.h"
#include "MRInplaceStack.h"
#include "MRParallelFor.h"
#include "MRTimer.h"
#include "MRPch/MRTBB.h"
#include <limits>

namespace MR
{

namespace
{

/// the squared distances from pt to the children of node, with xf applied to their boxes if given
template <bool Plain>
inline void childDistSq( const AABBTreePoints::SimdNode & node, const Vector3f & pt, const AffineXf3f * xf,
    Simd4::F px, Simd4::F py, Simd4::F pz, float ( &d )[4] )
{
    if constexpr ( !Plain )
    {
        if ( xf )
        {
            for ( int i = 0; i < AABBTreePoints::MaxChildren; ++i )
            {
                const auto box = node.get( i );
                d[i] = box.valid() ? transformed( box, *xf ).getDistanceSq( pt ) : std::numeric_limits<float>::infinity();
            }
            return;
        }
    }
    Simd4::store( d, Simd4::mul( Simd4::boxDistSq( node.box, px, py, pz ), Simd4::set1( cSimdDistLow ) ) );
}

/// visits the leaves of not-empty tree nearest first, while their boxes are closer than limit();
/// visitLeaf( leaf ) returns true to stop the search
template <bool Plain, typename L, typename V>
void forLeavesNearestFirst( const AABBTreePoints & tree, const Vector3f & pt, const AffineXf3f * xf, L && limit, V && visitLeaf )
{
    using namespace Simd4;
    const auto & simdNodes = tree.simdNodes();
    const F px = set1( pt.x ), py = set1( pt.y ), pz = set1( pt.z );
    // every level adds at most MaxChildren - 1 tasks
    InplaceStack<PointsSubTask, 3 * AABBTreePoints::MaxDepth + 1> subtasks;

    const auto rootBox = tree.getBoundingBox();
    PointsSubTask s{ tree.rootRef(), ( !Plain && xf ) ? transformed( rootBox, *xf ).getDistanceSq( pt ) : rootBox.getDistanceSq( pt ) };
    for (;;)
    {
        if ( s.distSq < limit() )
        {
            if ( s.ref < 0 )
            {
                if ( visitLeaf( ~s.ref ) )
                    return;
            }
            else
            {
                alignas( 16 ) float d[4];
                childDistSq<Plain>( simdNodes[s.ref], pt, xf, px, py, pz, d );
                PointsSubTask ch[4];
                const int numCh = sortedChildren( simdNodes[s.ref], d, limit(), ch );
                if ( numCh > 0 )
                {
                    // descend in the nearest child, the others wait in the stack
                    for ( int i = 0; i + 1 < numCh; ++i )
                        subtasks.push( ch[i] );
                    s = ch[numCh - 1];
                    continue;
                }
            }
        }
        if ( subtasks.empty() )
            return;
        s = subtasks.top();
        subtasks.pop();
    }
}

/// Plain: no transformation, region or skipCb, so the distances to the points of a leaf are found four at once
template <bool Plain>
PointsProjectionResult findProjectionOnPointsT( const Vector3f& pt, const AABBTreePoints& tree,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq, const VertBitSet* region, const VertPredicate & skipCb )
{
    PointsProjectionResult res;
    res.distSq = upDistLimitSq;
    if ( tree.empty() )
        return res;
    const auto& orderedPoints = tree.orderedPoints();
    const Simd4::F px = Simd4::set1( pt.x ), py = Simd4::set1( pt.y ), pz = Simd4::set1( pt.z );

    // returns true if the point is not farther than loDistLimitSq
    auto tryPoint = [&] ( int i, float distSq )
    {
        if ( !( distSq < res.distSq ) )
            return false;
        res.distSq = distSq;
        res.vId = orderedPoints[i].id;
        return distSq <= loDistLimitSq;
    };

    forLeavesNearestFirst<Plain>( tree, pt, xf, [&] { return res.distSq; }, [&] ( int leaf )
    {
        auto [first, last] = tree.leafPointRange( leaf );
        if constexpr ( Plain )
        {
            for ( ; first + 4 <= last; first += 4 )
            {
                const auto dv = Simd4::mul( pointsDistSq( orderedPoints.data() + first, px, py, pz ), Simd4::set1( cSimdDistLow ) );
                for ( int m = Simd4::bits( Simd4::lt( dv, Simd4::set1( res.distSq ) ) ); m; m &= m - 1 )
                {
                    const int i = first + Simd4::lowestBit( m );
                    if ( tryPoint( i, ( orderedPoints[i].coord - pt ).lengthSq() ) )
                        return true;
                }
            }
        }
        for ( int i = first; i < last; ++i )
        {
            if constexpr ( !Plain )
            {
                if ( region && !region->test( orderedPoints[i].id ) )
                    continue;
                if ( skipCb && skipCb( orderedPoints[i].id ) )
                    continue;
            }
            const auto proj = ( !Plain && xf ) ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
            if ( tryPoint( i, ( proj - pt ).lengthSq() ) )
                return true;
        }
        return false;
    } );
    return res;
}

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
    if ( !xf && !region && !skipCb )
        return findProjectionOnPointsT<true>( pt, tree, upDistLimitSq, xf, loDistLimitSq, region, skipCb );
    return findProjectionOnPointsT<false>( pt, tree, upDistLimitSq, xf, loDistLimitSq, region, skipCb );
}

void findFewClosestPoints( const Vector3f& pt, const PointCloud& pc, FewSmallest<PointsProjectionResult> & res,
    float upDistLimitSq, const AffineXf3f* xf, float loDistLimitSq )
{
    const auto& tree = pc.getAABBTree();
    // must come after getAABBTree() to avoid situations when the same thread executing getAABBTree enters recursively
    // in this function and appends to not-empty res
    res.clear();
    if ( tree.empty() )
        return;
    const auto& orderedPoints = tree.orderedPoints();
    const Simd4::F px = Simd4::set1( pt.x ), py = Simd4::set1( pt.y ), pz = Simd4::set1( pt.z );

    auto topDistSq = [&]
    {
        return !res.full() ? upDistLimitSq : res.top().distSq;
    };
    // returns true if the search can stop
    auto tryPoint = [&] ( int i, float distSq )
    {
        if ( !( distSq < topDistSq() ) )
            return false;
        res.push( { .distSq = distSq, .vId = orderedPoints[i].id } );
        return res.full() && res.top().distSq <= loDistLimitSq;
    };
    auto visitLeaf = [&] ( int leaf )
    {
        auto [first, last] = tree.leafPointRange( leaf );
        if ( !xf )
        {
            for ( ; first + 4 <= last; first += 4 )
            {
                const auto dv = Simd4::mul( pointsDistSq( orderedPoints.data() + first, px, py, pz ), Simd4::set1( cSimdDistLow ) );
                for ( int m = Simd4::bits( Simd4::lt( dv, Simd4::set1( topDistSq() ) ) ); m; m &= m - 1 )
                {
                    const int i = first + Simd4::lowestBit( m );
                    if ( tryPoint( i, ( orderedPoints[i].coord - pt ).lengthSq() ) )
                        return true;
                }
            }
        }
        for ( int i = first; i < last; ++i )
        {
            const auto proj = xf ? ( *xf )( orderedPoints[i].coord ) : orderedPoints[i].coord;
            if ( tryPoint( i, ( proj - pt ).lengthSq() ) )
                return true;
        }
        return false;
    };
    if ( xf )
        forLeavesNearestFirst<false>( tree, pt, xf, topDistSq, visitLeaf );
    else
        forLeavesNearestFirst<true>( tree, pt, xf, topDistSq, visitLeaf );
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
    const VertCoords& points, const FindProjectionOnPointsSettings& settings ) const
{
    if ( !pointCloud_ )
        return unexpected( "No reference point cloud is set" );

    /// force compute AABB tree
    (void)pointCloud_->getAABBTree();

    results.resize( points.size() );
    ParallelFor( points, [&] ( VertId i )
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
