#include "MRPointCloudTriangulationHelpers.h"
#include "MRMeshDelone.h"
#include "MRVector3.h"
#include "MRPointCloud.h"
#include "MRPointsInBall.h"
#include "MRBestFit.h"
#include "MRPlane3.h"
#include "MRTriMath.h"
#include "MRGeodesicPath.h"
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
bool flipPossibility( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d,
    const Vector3f& aNorm, const Vector3f& cNorm )
{
    if ( dot( aNorm, cNorm ) < 0.0f )
        return true;

    if ( !isUnfoldQuadrangleConvex( a, b, c, d ) )
        return false;
    
    return true;
}


float deloneFlipProfit( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d )
{
    auto metricAC = std::max( circumcircleDiameter( a, c, d ), circumcircleDiameter( c, a, b ) );
    auto metricBD = std::max( circumcircleDiameter( b, d, a ), circumcircleDiameter( d, b, c ) );
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

float updateNeighborsRadius( const VertCoords& points, VertId v, const std::vector<VertId>& fan, float baseRadius )
{
    float maxRadius = 0.0f;

    // increase radius if better local triangulation can exist
    for ( int i = 0; i < fan.size(); ++i )
    {
        auto next = cycleNext( fan, i );
        maxRadius = std::max( maxRadius, circumcircleDiameter( 
            points[v],
            points[fan[i]],
            points[fan[next]] ) );
    }

    return std::min( maxRadius, 2.0f * baseRadius );
}

void findNeighbors( const PointCloud& pointCloud, VertId v, float radius, std::vector<VertId>& neighbors )
{
    neighbors.clear();
    const auto& points = pointCloud.points;
    findPointsInBall( pointCloud, points[v], radius, [&]( VertId vid, const Vector3f& )
    {
        if ( vid != v )
            neighbors.push_back( vid );
    } );
}

struct FanOptimizerQueueElement
{
    float weight{ 0.0f }; // profit of flipping this edge
    int id{ -1 }; // index

    // needed to remove outdated queue elements
    int prevId{ -1 }; // id of prev neighbor
    int nextId{ -1 }; // id of next neighbor

    bool stable{ false }; // if this flag is true, edge cannot be flipped
    bool operator < ( const FanOptimizerQueueElement& other ) const
    {
        if ( stable == other.stable )
            return weight < other.weight;
        return stable;
    }
    bool operator==( const FanOptimizerQueueElement& other ) const = default;

    bool isOutdated( const std::vector<VertId>& neighbors ) const
    {
        return !neighbors[nextId].valid() || !neighbors[prevId].valid();
    }
};

class FanOptimizer
{
public:
    FanOptimizer( const VertCoords& points, const VertCoords& normals, TriangulatedFanData& fanData, VertId centerVert ):
        centerVert_{ centerVert },
        fanData_{ fanData },
        points_{ points },
        normals_{ normals }
    {
        init_();
    }
    void optimize( int steps, float critAngle );
    void updateBorder( float angle = MR::PI_F );
private:
    Plane3f plane_;

    VertId centerVert_;
    TriangulatedFanData& fanData_;
    const VertCoords& points_;
    const VertCoords& normals_;

    void init_();

    FanOptimizerQueueElement calcQueueElement_( const std::vector<VertId>& neighbors, int i, float critAngle ) const;
};


FanOptimizerQueueElement FanOptimizer::calcQueueElement_(
   const std::vector<VertId>& neighbors, int i, float critAngle ) const
{
    FanOptimizerQueueElement res;
    res.id = i;
    res.nextId = cycleNext( neighbors, i );
    res.prevId = cyclePrev( neighbors, i );
    const auto& a = points_[centerVert_];
    const auto& b = points_[neighbors[res.nextId]];
    const auto& c = points_[neighbors[res.id]];
    const auto& d = points_[neighbors[res.prevId]];

    const auto& aNorm = normals_[centerVert_];
    const auto& cNorm = normals_[neighbors[res.id]];

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
    res.stable = !( flipPossibility( a, b, c, d, aNorm, cNorm ) && ( deloneProf > 0.0f || angleProf > 0.0f ) );
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

void FanOptimizer::optimize( int steps, float critAng )
{
    updateBorder();
    if ( steps == 0 )
        return;

    std::priority_queue<FanOptimizerQueueElement> queue_;
    for ( int i = 0; i < fanData_.neighbors.size(); ++i )
    {
        auto el = calcQueueElement_( fanData_.neighbors, i, critAng );
        if ( fanData_.border == fanData_.neighbors[i] )
            el.stable = true;
        queue_.push( el );
    }

    // optimize fan
    int allRemoves = 0;
    int currentFanSize = int( fanData_.neighbors.size() );
    while ( !queue_.empty() )
    {
        auto topEl = queue_.top();
        queue_.pop();
        if ( !fanData_.neighbors[topEl.id].valid() )
            continue; // this vert was erased
        if ( topEl.isOutdated( fanData_.neighbors ) )
            continue; // topEl is not valid, because its neighbor was erased
        if ( topEl.stable )
            break; // topEl valid and fan is stable

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
        queue_.emplace( calcQueueElement_( fanData_.neighbors, topEl.nextId, critAng ) );
        queue_.emplace( calcQueueElement_( fanData_.neighbors, topEl.prevId, critAng ) );
    }

    fanData_.neighbors.erase(
        std::remove_if( fanData_.neighbors.begin(), fanData_.neighbors.end(), [] ( VertId  v )
    {
        return !v.valid();
    } ),
        fanData_.neighbors.end() );
}

void FanOptimizer::updateBorder( float angle )
{
    for ( int i = 0; i < fanData_.cacheAngleOrder.size(); ++i )
    {
        // check border fans
        auto diff = ( i + 1 < fanData_.cacheAngleOrder.size() ) ?
            ( fanData_.cacheAngleOrder[i + 1].first - fanData_.cacheAngleOrder[i].first ) :
            ( fanData_.cacheAngleOrder[0].first + 2.0 * PI - fanData_.cacheAngleOrder[i].first );
        if ( diff > angle )
            fanData_.border = fanData_.neighbors[i];
    }
}

void FanOptimizer::init_()
{
    Vector3f centerProj = points_[centerVert_];
    plane_ = Plane3f::fromDirAndPt( normals_[centerVert_], centerProj );

    Vector3f firstProj = plane_.project( points_[fanData_.neighbors.front()] );
    Vector3f baseVec = ( firstProj - centerProj ).normalized();

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

void trianglulateFan( const VertCoords& points, VertId centerVert, TriangulatedFanData& triangulationData,
    const VertCoords& normals, float critAngle, int steps /*= INT_MAX */ )
{
    if ( triangulationData.neighbors.empty() )
        return;
    FanOptimizer optimizer( points, normals, triangulationData, centerVert );
    optimizer.optimize( steps, critAngle );
}

void buildLocalTriangulation( const PointCloud& cloud, VertId v, const VertCoords& normals, const Settings & settings,
    TriangulatedFanData & fanData )
{
    findNeighbors( cloud, v, settings.radius, fanData.neighbors );
    trianglulateFan( cloud.points, v, fanData, normals, settings.critAngle );

    float maxRadius = ( fanData.neighbors.size() < 2 ) ? settings.radius * 2 :
        updateNeighborsRadius( cloud.points, v, fanData.neighbors, settings.radius );

    if ( maxRadius > settings.radius )
    {
        // update triangulation if radius was increased
        findNeighbors( cloud, v, maxRadius, fanData.neighbors );
        trianglulateFan( cloud.points, v, fanData, normals, settings.critAngle );
    }
}

bool isBoundaryPoint( const PointCloud& pointCloud, const VertCoords& normals, 
    VertId v, float radius, float angle, TriangulatedFanData& triangulationData )
{
    TriangulationHelpers::findNeighbors( pointCloud, v, radius, triangulationData.neighbors );
    float maxRadius = ( triangulationData.neighbors.size() < 2 ) ? radius * 2.0f :
        TriangulationHelpers::updateNeighborsRadius( pointCloud.points, v, triangulationData.neighbors, radius );
    if ( maxRadius > radius )
        TriangulationHelpers::findNeighbors( pointCloud, v, maxRadius, triangulationData.neighbors );
    
    triangulationData.border = {};
    if ( triangulationData.neighbors.size() < 3 )
        return true;
    FanOptimizer optimizer( pointCloud.points, normals, triangulationData, v );
    optimizer.updateBorder( angle );
    return triangulationData.border.valid();
}

} //namespace TriangulationHelpers
} //namespace MR
