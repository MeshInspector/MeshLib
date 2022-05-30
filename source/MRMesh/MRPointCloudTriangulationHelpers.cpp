#include "MRPointCloudTriangulationHelpers.h"
#include "MRMeshDelone.h"
#include "MRVector3.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRPlane3.h"
#include <algorithm>
#include <queue>
#include <numeric>
#include <limits>

namespace MR
{

namespace
{

// this function check that abc acd is allowed to be flipped to abd dbc
// aNorm - normal in point a
// cNorm - normal in point c
// planeDist - distance from c to plane (a,aNorm)
bool flipPossibility( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d,
    const Vector3f& aNorm, const Vector3f& cNorm,
    float planeDist )
{
    if ( dot( aNorm, cNorm ) < 0.0f )
        return true;

    if ( planeDist * planeDist > ( b - d ).lengthSq() )
        return true;

    if ( ( c - a ).lengthSq() < ( 0.5f * ( b + d ) - a ).lengthSq() )
        return false;
    return true;
}


float deloneFlipProfit( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    auto metricAC = std::max( circumcircleDiameter( a, c, d ), circumcircleDiameter( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameter( b, d, a ), circumcircleDiameter( d, b, c ) );
    return metricAC - metricBD;
}

// check that edge angle is less then critical, and C point is further than B and D
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
template<typename T>
typename std::list<T>::const_iterator cycleNext( const std::list<T>& list, const typename std::list<T>::const_iterator& it )
{
    if ( std::next( it ) == list.end() )
        return list.begin();
    return std::next( it );
}

template<typename T>
typename std::list<T>::const_iterator cyclePrev( const std::list<T>& list, const typename std::list<T>::const_iterator& it )
{
    if ( it == list.begin() )
        return std::prev( list.end() );
    return std::prev( it );
}

float updateNeighborsRadius( const VertCoords& points, VertId v, const std::list<VertId>& fan, float baseRadius )
{
    float maxRadius = 0.0f;

    // increase radius if better local triangulation can exist
    for ( auto it = fan.begin(); it != fan.end(); ++it )
    {
        auto next = cycleNext( fan, it );
        maxRadius = std::max( maxRadius, circumcircleDiameter( 
            points[v],
            points[*it],
            points[*next] ) );
    }

    return std::min( maxRadius, 2.0f * baseRadius );
}

std::vector<VertId> findNeighbors( const PointCloud& pointCloud, VertId v, float radius )
{
    std::vector<VertId> res;
    const auto& points = pointCloud.points;
    findPointsInBall( pointCloud, points[v], radius, [&]( VertId vid, const Vector3f& )
    {
        if ( vid != v )
            res.push_back( vid );
    } );
    return res;
}

struct FanOptimizerQueueElement
{
    float weight{ 0.0f }; // profit of flipping this edge
    std::list<int>::const_iterator pIt; // iterator to get neighbors
    int id{ -1 }; // id in FanOptimizer::angleOrder_, to check if this element is still present in FanOptimizer::presentNeighbors_
    bool stable{ false }; // if this flag is true, edge cannot be flipped
    bool operator < ( const FanOptimizerQueueElement& other ) const
    {
        if ( stable == other.stable )
            return weight < other.weight;
        return stable;
    }
    bool operator==( const FanOptimizerQueueElement& other ) const = default;
};

class FanOptimizer
{
public:
    FanOptimizer( const VertCoords& points, const VertCoords& normals, const std::vector<VertId>& neighbors, VertId centerVert ):
        centerVert_{ centerVert },
        neighbors_{neighbors},
        points_{ points },
        normals_{ normals }
    {
        init_();
    }
    TriangulatedFan optimize( int steps, float critAngle );
private:
    std::vector<std::pair<double, int>> angleOrder_;
    BitSet presentNeighbors_;
    Plane3f plane_;

    VertId centerVert_;
    const std::vector<VertId>& neighbors_;
    const VertCoords& points_;
    const VertCoords& normals_;

    void init_();
    VertId getVertByPos_( int i ) const;

    FanOptimizerQueueElement calcQueueElement_( 
        const std::list<int>& list, const std::list<int>::const_iterator& it,
        float critAngle ) const;
};


FanOptimizerQueueElement FanOptimizer::calcQueueElement_(
    const std::list<int>& list, const std::list<int>::const_iterator& it, float critAngle ) const
{
    FanOptimizerQueueElement res;
    res.pIt = it;
    res.id = *it;
    const auto& a = points_[centerVert_];
    const auto& b = points_[getVertByPos_( *cycleNext( list, it ) )];
    const auto& c = points_[getVertByPos_( *it )];
    const auto& d = points_[getVertByPos_( *cyclePrev( list, it ) )];

    const auto& aNorm = normals_[centerVert_];
    const auto& cNorm = normals_[getVertByPos_( *it )];

    float normVal = ( c - a ).length();
    if ( normVal == 0.0f )
    {
        res.weight = std::numeric_limits<float>::max();
        return res;
    }
    float planeDist = std::abs( plane_.distance( c ) );
    auto deloneProf = deloneFlipProfit( a, b, c, d ) / normVal;
    auto angleProf = trisAngleProfit( a, b, c, d, critAngle );
    // ( deloneProf > 0.0f || angleProf > 0.0f )  strict condition to have more faces options if flip is not profitable
    res.stable = !( flipPossibility( a, b, c, d, aNorm, cNorm, planeDist ) && ( deloneProf > 0.0f || angleProf > 0.0f ) );
    if ( deloneProf > 0.0f )
        res.weight += deloneProf;
    if ( angleProf > 0.0f )
        res.weight += angleProf;

    res.weight += planeDist / normVal;

    res.weight += 5.0f * ( 1.0f - dot( normals_[centerVert_], cNorm ) );

    auto abcNorm = cross( b - a, c - a );
    auto acdNorm = cross( c - a, d - a );

    auto triNormWeight = dot( ( abcNorm + acdNorm ).normalized(), cNorm );
    if ( triNormWeight < 0.0f )
        res.weight = std::numeric_limits<float>::max();
    else
        res.weight += 5.0f * ( 1.0f - triNormWeight );

    return res;
}

TriangulatedFan FanOptimizer::optimize( int steps, float critAng )
{
    TriangulatedFan res;
    res.optimized.resize( angleOrder_.size() );
    int i = 0;
    for ( auto it = res.optimized.begin(); it != res.optimized.end(); ++it, ++i )
    {
        ( *it ) = getVertByPos_( i );
        // check border fans
        auto diff = ( i + 1 < angleOrder_.size() ) ? 
            ( angleOrder_[i + 1].first - angleOrder_[i].first ) : 
            ( angleOrder_[0].first + 2.0 * PI - angleOrder_[i].first );
        if ( diff > PI )
            res.border = *it;
    }
    if ( steps == 0 )
        return res;

    res.optimized.clear();
    std::list<int> posList( angleOrder_.size() );
    std::iota( posList.begin(), posList.end(), 0 );
    presentNeighbors_.resize( posList.size(), true );

    std::priority_queue<FanOptimizerQueueElement> queue_;
    for ( auto it = posList.begin(); it != posList.end(); ++it )
    {
        auto el = calcQueueElement_( posList, it, critAng );
        if ( res.border == getVertByPos_( *it ) )
            el.stable = true;
        queue_.push( el );
    }

    // optimize fan
    int allRemoves = 0;
    while ( !queue_.empty() )
    {
        auto topEl = queue_.top();
        queue_.pop();
        if ( !presentNeighbors_.test( topEl.id ) )
            continue; // this vert was erased
        auto recalcEl = calcQueueElement_( posList, topEl.pIt, critAng );
        if ( topEl != recalcEl )
            continue; // topEl is not valid, because its neighbor was erased
        if ( topEl.stable )
            break; // topEl valid and fan is stable
        auto left = cycleNext( posList, topEl.pIt );
        auto right = cyclePrev( posList, topEl.pIt );
        posList.erase( topEl.pIt );
        presentNeighbors_.reset( topEl.id );
        allRemoves++;
        if ( allRemoves >= steps )
            break;
        if ( posList.size() < 2 )
        {
            posList.clear();
            break;
        }
        auto leftEl = calcQueueElement_( posList, left, critAng );
        auto rightEl = calcQueueElement_( posList, right, critAng );
        queue_.push( leftEl );
        queue_.push( rightEl );
    }

    res.optimized.resize( posList.size() );
    auto resIt = res.optimized.begin();
    for ( auto it = posList.begin(); it != posList.end(); ++it, ++resIt )
        ( *resIt ) = getVertByPos_( *it );
    return res;

}

void FanOptimizer::init_()
{
    Vector3f centerProj = points_[centerVert_];
    plane_ = Plane3f::fromDirAndPt( normals_[centerVert_], centerProj );

    Vector3f firstProj = plane_.project( points_[neighbors_.front()] );
    Vector3f baseVec = ( firstProj - centerProj ).normalized();

    // fill angles
    angleOrder_.resize( neighbors_.size() );
    for ( int i = 0; i < neighbors_.size(); ++i )
    {
        auto vec = ( plane_.project( points_[neighbors_[i]] ) - centerProj ).normalized();
        auto crossProd = cross( vec, baseVec );
        double sign = 1.0;
        if ( dot( crossProd, plane_.n ) < 0 )
            sign = -1.0;
        angleOrder_[i] = { std::atan2( sign * crossProd.length() ,dot( vec,baseVec ) ),i };
    }

    // sort candidates
    std::sort( angleOrder_.begin(), angleOrder_.end() );
}

VertId FanOptimizer::getVertByPos_( int i ) const
{
    return neighbors_[angleOrder_[i].second];
}

TriangulatedFan trianglulateFan( const VertCoords& points, VertId centerVert, const std::vector<VertId>& neighbors, 
    const VertCoords& normals, float critAngle, int steps /*= INT_MAX */ )
{
    if ( neighbors.empty() )
        return {};
    FanOptimizer optimizer( points, normals, neighbors, centerVert );
    return optimizer.optimize( steps, critAngle );
}

} //namespace TriangulationHelpers
} //namespace MR
