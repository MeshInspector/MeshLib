#include "MRPointCloudTriangulationHelpers.h"
#include "MRMeshDelone.h"
#include "MRVector3.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRTriMath.h"
#include "MRGeodesicPath.h"
#include "MRTimer.h"
#include "MRBitSetParallelFor.h"
#include "MRLocalTriangulations.h"
#include <algorithm>
#include <numeric>
#include <limits>

namespace MR
{

namespace
{

float deloneFlipProfitSq( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    auto metricAC = std::max( circumcircleDiameterSq( a, c, d ), circumcircleDiameterSq( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameterSq( b, d, a ), circumcircleDiameterSq( d, b, c ) );
    return metricAC - metricBD;
}

// check that edge angle is less then critical
float trisAngleProfit( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float critAng )
{
    auto ac = ( c - a );
    auto ab = ( b - a );
    auto ad = ( d - a );

    auto dirABC = cross( ab, ac );
    auto dirACD = cross( ac, ad );

    return angle( dirABC, dirACD ) - critAng;
}
}

namespace TriangulationHelpers
{
int cycleNext( const std::vector<VertId>& neighbors, int i )
{
    for ( ;;)
    {
        ++i;
        if ( i == neighbors.size() )
            i = 0;
        if ( neighbors[i].valid() )
            return i;
    }
}

int cyclePrev( const std::vector<VertId>& neighbors, int i )
{
    for ( ;;)
    {
        --i;
        if ( i == -1 )
            i = int( neighbors.size() ) - 1;
        if ( neighbors[i].valid() )
            return i;
    }
}

float updateNeighborsRadius( const VertCoords& points, VertId v, VertId borderV,
    const std::vector<VertId>& fan, float baseRadius )
{
    float maxRadius = 0.0f;

    // increase radius if better local triangulation can exist
    for ( int i = 0; i < fan.size(); ++i )
    {
        auto next = cycleNext( fan, i );
        if ( fan[i] == borderV )
            continue;

        const auto & a = points[v];
        const auto & b = points[fan[i]];
        const auto & c = points[fan[next]];
        const auto cdSq = circumcircleDiameterSq( a, b, c );
        if ( sqr( maxRadius ) >= cdSq )
            continue;

        const auto cc = circumcircleCenter( b - a, c - a );

        const auto cl = cc.length(); // distance between point[v] and the center of circumcicle
        const auto cr = 0.5f * std::sqrt( cdSq ); // radius
        // the center of circumcicle must be within its radius from point[v]
        // assert( cl <= cr ); may be wrong due to floating-point errors

        // circumcicle must be fully within the ball around point[v]
        maxRadius = std::max( maxRadius, cl + cr );
    }

    return std::min( maxRadius, 2.0f * baseRadius );
}

void findNeighborsInBall( const PointCloud& pointCloud, VertId v, float radius, std::vector<VertId>& neighbors )
{
    neighbors.clear();
    const auto& points = pointCloud.points;
    findPointsInBall( pointCloud, points[v], radius, [&]( VertId vid, const Vector3f& )
    {
        if ( vid != v )
            neighbors.push_back( vid );
    } );
}

float findNumNeighbors( const PointCloud& pointCloud, VertId v, int numNeis, std::vector<VertId>& neighbors,
    FewSmallest<PointsProjectionResult> & tmp, float upDistLimitSq )
{
    tmp.reset( numNeis + 1 );
    findFewClosestPoints( pointCloud.points[v], pointCloud, tmp, upDistLimitSq );
    auto maxDistSq = tmp.empty() ? 0.0f : tmp.top().distSq;
    neighbors.clear();
    for ( const auto & n : tmp.get() )
        if ( n.vId != v )
            neighbors.push_back( n.vId );
    return maxDistSq;
}

void filterNeighbors( const VertNormals& normals, VertId v, std::vector<VertId>& neighbors )
{
    const auto& vNorm = normals[v];
    neighbors.erase( std::remove_if( neighbors.begin(), neighbors.end(), [&] ( VertId nv )
    {
        return dot( vNorm, normals[nv] ) < -0.3f;
    } ), neighbors.end() );
}

class FanOptimizer
{
public:
    FanOptimizer( const VertCoords& points, const VertCoords* trustedNormals, TriangulatedFanData& fanData, VertId centerVert, const PointCloud * searchCloud, float maxEdgeLen ):
        centerVert_{ centerVert },
        fanData_{ fanData },
        points_{ points },
        trustedNormals_{ trustedNormals },
        searchCloud_{ searchCloud },
        maxEdgeLenSq_{ sqr( maxEdgeLen ) }
    {
        init_();
    }
    void optimize( int steps, float critAngle, float boundaryAngle );
    void updateBorder( float angle );
private:
    Plane3f plane_;
    float normalizerSq_{ 0.0f };

    VertId centerVert_;
    TriangulatedFanData& fanData_;
    const VertCoords& points_;
    const VertCoords* trustedNormals_ = nullptr;

    const PointCloud * searchCloud_ = nullptr;
    float maxEdgeLenSq_ = 0;

    void init_();

    // faces with aspect ratio more than this is optimized first
    static constexpr float CriticalAspectRatio = 1e3f;
    FanOptimizerQueueElement calcQueueElement_( int i, float critAngle ) const;

    void updateBorderQueueElement_( FanOptimizerQueueElement& res, bool nextEl ) const;
};


FanOptimizerQueueElement FanOptimizer::calcQueueElement_( int i, float critAngle ) const
{
    FanOptimizerQueueElement res;
    res.id = i;
    res.nextId = cycleNext( fanData_.neighbors, i );
    res.prevId = cyclePrev( fanData_.neighbors, i );

    if ( fanData_.border == fanData_.neighbors[res.id] )
    {
        updateBorderQueueElement_( res, false );
        return res;
    }
    else if ( fanData_.border == fanData_.neighbors[res.prevId] )
    {
        updateBorderQueueElement_( res, true );
        return res;
    }

    auto difAngle = fanData_.cacheAngleOrder[res.nextId].first - fanData_.cacheAngleOrder[res.prevId].first;
    if ( difAngle < 0.0 )
        difAngle += 2.0 * PI;
    if ( difAngle > PI )
    {
        // prohibit removal of this edge since the "angle" of remaining triangle will be more than PI
        res.stable = true;
        return res;
    }

    const auto av = centerVert_;
    const auto bv = fanData_.neighbors[res.nextId];
    const auto cv = fanData_.neighbors[res.id];
    const auto dv = fanData_.neighbors[res.prevId];

    if ( searchCloud_  && ( searchCloud_->points[bv] - searchCloud_->points[dv] ).lengthSq() > maxEdgeLenSq_ )
    {
        // prohibit removal of this edge since newly appeared edge will be too long
        res.stable = true;
        return res;
    }

    const auto& a = points_[av];
    const auto& b = points_[bv];
    const auto& c = points_[cv];
    const auto& d = points_[dv];

    auto acLengthSq = ( a - c ).lengthSq();
    if ( ( acLengthSq > ( b - a ).lengthSq() && triangleAspectRatio( a, b, c ) > CriticalAspectRatio ) ||
        ( acLengthSq > ( d - a ).lengthSq() && triangleAspectRatio( a, c, d ) > CriticalAspectRatio ) )
    {
        // this edge belongs to a degenerate triangle and it is the longest, remove it is fast as possible
        res.weight = FLT_MAX;
        return res;
    }

    // whether abc acd is allowed to be flipped to abd dbc
    bool flipPossibility = false;
    if ( trustedNormals_ && ( dot( (*trustedNormals_)[centerVert_], (*trustedNormals_)[fanData_.neighbors[res.id]] ) < 0.0f ) )
        flipPossibility = true;
    else
        flipPossibility = isUnfoldQuadrangleConvex( a, b, c, d );

    if ( !flipPossibility )
    {
        // removal of AC is prohibited
        res.stable = true;
        return res;
    }

    auto deloneProf = deloneFlipProfitSq( a, b, c, d );
    // deloneProf == 0 iff all 4 points are located exactly on a circle;
    // select the edge with minimal vertex indices to break the tie
    if ( deloneProf == 0 && std::min( av, cv ) > std::min( bv, dv ) )
        deloneProf = -1;
    auto angleProf = trisAngleProfit( a, b, c, d, critAngle );
    if ( deloneProf < 0 && angleProf <= 0 )
    {
        // removal of AC is prohibited
        res.stable = true;
        return res;
    }

    if ( deloneProf > 0.0f )
        res.weight += deloneProf / normalizerSq_;
    if ( angleProf > 0.0f )
        res.weight += angleProf;

    float normVal = ( c - a ).length();
    if ( normVal == 0.0f )
    {
        // a neighbor point coincides with this one => remove it with maximal weight
        assert( false ); // all such points must be deleted before
        res.weight = FLT_MAX;
        return res;
    }
    float planeDist = std::abs( plane_.distance( c ) );
    res.weight += planeDist / normVal; // sin angle with plane

    if ( trustedNormals_ )
    {
        const auto cNorm = (*trustedNormals_)[fanData_.neighbors[res.id]];
        res.weight += 5.0f * ( 1.0f - dot( (*trustedNormals_)[centerVert_], cNorm ) );

        auto abcNorm = cross( b - a, c - a );
        auto acdNorm = cross( c - a, d - a );

        auto triNormWeight = dot( ( abcNorm + acdNorm ).normalized(), cNorm );
        if ( triNormWeight < 0.0f )
            res.weight = std::numeric_limits<float>::max();
        else
            res.weight += 5.0f * ( 1.0f - triNormWeight );
    }

    return res;
}

void FanOptimizer::updateBorderQueueElement_( FanOptimizerQueueElement& res, bool nextEl ) const
{
    assert( nextEl ? fanData_.border == fanData_.neighbors[res.prevId] : fanData_.border == fanData_.neighbors[res.id] );

    int prevInd = nextEl ? res.id : res.prevId;
    int nextInd = nextEl ? res.nextId : res.id;

    int otherId = nextEl ? res.nextId : res.prevId;
    auto lengthSq = ( points_[centerVert_] - points_[fanData_.neighbors[res.id]] ).lengthSq();
    auto otherLengthSq = ( points_[centerVert_] - points_[fanData_.neighbors[otherId]] ).lengthSq();
    if ( lengthSq < otherLengthSq )
    {
        // prohibit removal of this boundary edge since its neighbor edge is longer
        res.stable = true;
        return;
    }

    const auto av = centerVert_;
    const auto bv = fanData_.neighbors[prevInd];
    const auto cv = fanData_.neighbors[nextInd];

    const auto& a = points_[av];
    const auto& b = points_[bv];
    const auto& c = points_[cv];

    if ( triangleAspectRatio( a, b, c ) <= CriticalAspectRatio )
    {
        // prohibit removal of this boundary edge since the boundary triangle is not degenerate
        res.stable = true;
        return;
    }

    // remove this boundary edge as fast as possible
    res.weight = std::numeric_limits<float>::max();
}

void FanOptimizer::optimize( int steps, float critAng, float boundaryAngle )
{
    updateBorder( boundaryAngle );
    if ( steps == 0 )
        return;

    auto & queue = fanData_.queue;
    while ( !queue.empty() )
        queue.pop();

    int currentFanSize = int( fanData_.neighbors.size() );
    for ( int i = 0; i < fanData_.neighbors.size(); ++i )
    {
        if ( points_[fanData_.neighbors[i]] == points_[centerVert_] )
        {
            fanData_.neighbors[i] = {}; // remove points coinciding with center one
            --currentFanSize;
        }
        else if ( auto x = calcQueueElement_( i, critAng ); !x.stable )
            queue.emplace( std::move( x ) );
    }
    if ( currentFanSize < 2 )
    {
        fanData_.neighbors.clear();
        return;
    }

    // optimize fan
    int allRemoves = 0;
    while ( !queue.empty() )
    {
        auto topEl = queue.top();
        assert( !topEl.stable );
        queue.pop();
        if ( !fanData_.neighbors[topEl.id].valid() )
            continue; // this vert was erased
        if ( topEl.isOutdated( fanData_.neighbors ) )
            continue; // topEl is not valid, because its neighbor was erased

        auto oldNei = fanData_.neighbors[topEl.id];
        fanData_.neighbors[topEl.id] = {};
        allRemoves++;
        currentFanSize--;
        if ( allRemoves >= steps )
            break;
        if ( currentFanSize < 2 )
        {
            fanData_.neighbors.clear();
            return;
        }
        if ( oldNei == fanData_.border )
            fanData_.border = fanData_.neighbors[topEl.prevId];

        if ( auto x = calcQueueElement_( topEl.nextId, critAng ); !x.stable )
            queue.emplace( std::move( x ) );

        if ( auto x = calcQueueElement_( topEl.prevId, critAng ); !x.stable )
            queue.emplace( std::move( x ) );
    }

    erase_if( fanData_.neighbors, []( VertId v ) { return !v.valid(); } );
    if ( fanData_.neighbors.size() < 2 )
        fanData_.neighbors.clear();
}

void FanOptimizer::updateBorder( float angle )
{
    fanData_.border = {};
    for ( int i = 0; i < fanData_.cacheAngleOrder.size(); ++i )
    {
        // check border fans
        auto diff = ( i + 1 < fanData_.cacheAngleOrder.size() ) ?
            ( fanData_.cacheAngleOrder[i + 1].first - fanData_.cacheAngleOrder[i].first ) :
            ( fanData_.cacheAngleOrder[0].first + 2.0 * PI - fanData_.cacheAngleOrder[i].first );
        if ( diff > angle )
        {
            fanData_.border = fanData_.neighbors[i];
            break;
        }
    }
}

void FanOptimizer::init_()
{
    const Vector3f centerProj = points_[centerVert_];
    Vector3f centerNorm;
    if ( trustedNormals_ )
        centerNorm = (*trustedNormals_)[centerVert_];
    else
    {
        PointAccumulator accum;
        accum.addPoint( centerProj );
        for ( auto nid : fanData_.neighbors )
            accum.addPoint( points_[nid] );
        centerNorm = Vector3f( accum.getBestPlane().n );
    }
    plane_ = Plane3f::fromDirAndPt( centerNorm, centerProj );

    Vector3f firstProj = plane_.project( points_[fanData_.neighbors.front()] );
    Vector3f baseVec = ( firstProj - centerProj );
    normalizerSq_ = baseVec.lengthSq();
    if ( normalizerSq_ > 0.0f )
        baseVec = baseVec / std::sqrt( normalizerSq_ );
    else
    {
        baseVec = Vector3f();
        for ( int i = 1; i < fanData_.neighbors.size() && normalizerSq_ <= 0.0f; ++i )
        {
            auto proj = plane_.project( points_[fanData_.neighbors[i]] );
            normalizerSq_ = ( proj - centerProj ).lengthSq();
        }
        if ( normalizerSq_ <= 0.0f )
            normalizerSq_ = 1.0f; // all neighbors have same coordinate as center point
    }
    

    // fill angles
    fanData_.cacheAngleOrder.resize( fanData_.neighbors.size() );
    for ( int i = 0; i < fanData_.neighbors.size(); ++i )
    {
        auto vec = ( plane_.project( points_[fanData_.neighbors[i]] ) - centerProj ).normalized();
        auto crossProd = cross( vec, baseVec );
        double sign = 1.0;
        if ( dot( crossProd, plane_.n ) < 0 )
            sign = -1.0;
        fanData_.cacheAngleOrder[i] = { std::atan2( sign * crossProd.length() ,dot( vec,baseVec ) ),i };
    }

    // sort candidates
    std::sort( fanData_.cacheAngleOrder.begin(), fanData_.cacheAngleOrder.end() );
    for ( int i = 0; i < fanData_.neighbors.size(); ++i )
    {
        if ( fanData_.cacheAngleOrder[i].second == i )
            continue;
        int prevIndex = i;
        int nextIndex = fanData_.cacheAngleOrder[i].second;
        for ( ;;)
        {
            std::swap( fanData_.neighbors[prevIndex], fanData_.neighbors[nextIndex] );
            fanData_.cacheAngleOrder[prevIndex].second = prevIndex;
            prevIndex = nextIndex;
            nextIndex = fanData_.cacheAngleOrder[nextIndex].second;
            if ( nextIndex == i )
            {
                fanData_.cacheAngleOrder[prevIndex].second = prevIndex;
                break;
            }
        }
    }
}

static void trianglulateFan( const VertCoords& points, VertId centerVert, TriangulatedFanData& triangulationData,
    const Settings & settings )
{
    if ( triangulationData.neighbors.empty() )
        return;
    FanOptimizer optimizer( points, settings.trustedNormals, triangulationData, centerVert,
        settings.radius > 0 && !settings.automaticRadiusIncrease ? settings.searchNeighbors : nullptr,
        settings.radius );
    optimizer.optimize( settings.maxRemoves, settings.critAngle, settings.boundaryAngle );
    assert( triangulationData.neighbors.empty() || triangulationData.neighbors.size() > 1 );
}

void buildLocalTriangulation( const PointCloud& cloud, VertId v, const Settings & settings,
    TriangulatedFanData & fanData )
{
    float actualRadius = settings.radius;
    assert( ( settings.radius > 0 && settings.numNeis == 0 )
         || ( settings.radius == 0 && settings.numNeis > 0 ) );

    const auto & searchCloud = settings.searchNeighbors ? *settings.searchNeighbors : cloud;

    if ( settings.radius > 0 )
        findNeighborsInBall( searchCloud, v, actualRadius, fanData.neighbors );
    else
        actualRadius = std::sqrt( findNumNeighbors( searchCloud, v, settings.numNeis, fanData.neighbors, fanData.nearesetPoints ) );

    if ( settings.trustedNormals )
        filterNeighbors( *settings.trustedNormals, v, fanData.neighbors );
    if ( settings.allNeighbors )
        *settings.allNeighbors = fanData.neighbors;
    trianglulateFan( cloud.points, v, fanData, settings );

    if ( settings.automaticRadiusIncrease && actualRadius > 0 )
    {
        // if triangulation in original radius has border then we increase radius as well to find more neighbours
        float maxRadius = ( fanData.neighbors.size() < 2 || fanData.border ) ? actualRadius * 2 :
            updateNeighborsRadius( cloud.points, v, fanData.border, fanData.neighbors, actualRadius );

        if ( maxRadius > actualRadius )
        {
            // update triangulation if radius was increased
            actualRadius = maxRadius;
            if ( settings.radius > 0 )
                findNeighborsInBall( searchCloud, v, actualRadius, fanData.neighbors );
            else
            {
                // if the center point is an outlier then there may be too many points withing the ball of maxRadius;
                // so limit the search both by radius and by the number of neighbours
                actualRadius = std::sqrt( findNumNeighbors( searchCloud, v, std::max( 2 * settings.numNeis, 100 ),
                    fanData.neighbors, fanData.nearesetPoints, sqr( maxRadius ) ) );
            }

            if ( settings.trustedNormals )
                filterNeighbors( *settings.trustedNormals, v, fanData.neighbors );
            if ( settings.allNeighbors )
                *settings.allNeighbors = fanData.neighbors;
            trianglulateFan( cloud.points, v, fanData, settings );
        }
    }
    if ( settings.actualRadius )
        *settings.actualRadius = actualRadius;
}

std::optional<std::vector<SomeLocalTriangulations>> buildLocalTriangulations(
    const PointCloud& cloud, const Settings & settings, const ProgressCallback & progress )
{
    MR_TIMER

    // construct tree before parallel region
    if ( settings.searchNeighbors )
        settings.searchNeighbors->getAABBTree();
    else
        cloud.getAABBTree();

    struct PerThreadData : SomeLocalTriangulations
    {
        TriangulationHelpers::TriangulatedFanData fanData;
    };
    tbb::enumerable_thread_specific<PerThreadData> threadData;

    if ( !BitSetParallelFor( cloud.validPoints, [&]( VertId v )
    {
        auto& localData = threadData.local();
        auto& disc = localData.fanData;
        TriangulationHelpers::buildLocalTriangulation( cloud, v, settings, disc );

        localData.fanRecords.push_back( { v, disc.border, (std::uint32_t)localData.neighbors.size() } );
        localData.neighbors.insert( localData.neighbors.end(), disc.neighbors.begin(), disc.neighbors.end() );
        localData.maxCenterId = std::max( localData.maxCenterId, v );
    }, progress ) )
        return {};

    std::vector<SomeLocalTriangulations> res;
    res.reserve( threadData.size() );
    for ( auto & td : threadData )
    {
        td.fanRecords.push_back( { {}, {}, (std::uint32_t)td.neighbors.size() } );
        res.push_back( std::move( td ) );
    }

    return res;
}

std::optional<AllLocalTriangulations> buildUnitedLocalTriangulations(
    const PointCloud& cloud, const Settings & settings, const ProgressCallback & progress )
{
    MR_TIMER

    const auto optPerThreadTriangs = buildLocalTriangulations( cloud, settings, subprogress( progress, 0.0f, 0.9f ) );
    if ( !optPerThreadTriangs )
        return {};

    return uniteLocalTriangulations( *optPerThreadTriangs );
}

bool isBoundaryPoint( const PointCloud& cloud, VertId v, const Settings & settings,
    TriangulatedFanData & fanData )
{
    buildLocalTriangulation( cloud, v, settings, fanData );
    return fanData.border.valid();
}

std::optional<VertBitSet> findBoundaryPoints( const PointCloud& pointCloud, const Settings & settings,
    ProgressCallback cb )
{
    MR_TIMER

    VertBitSet borderPoints( pointCloud.validPoints.size() );
    tbb::enumerable_thread_specific<TriangulatedFanData> tls;
    if ( !BitSetParallelFor( pointCloud.validPoints, [&] ( VertId v )
    {
        auto& fanData = tls.local();
        if ( isBoundaryPoint( pointCloud, v, settings, fanData ) )
            borderPoints.set( v );
    }, cb ) )
        return {};
    return borderPoints;
}

} //namespace TriangulationHelpers

} //namespace MR
