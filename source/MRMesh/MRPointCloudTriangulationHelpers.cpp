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

void filterNeighbors( const VertNormals& normals, VertId v, std::vector<VertId>& neighbors )
{
    const auto& vNorm = normals[v];
    neighbors.erase( std::remove_if( neighbors.begin(), neighbors.end(), [&] ( VertId nv )
    {
        return dot( vNorm, normals[nv] ) < -0.3f;
    } ), neighbors.end() );
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
    FanOptimizer( const VertCoords& points, const VertCoords& normals, TriangulatedFanData& fanData, VertId centerVert, bool useNeiNormals ):
        centerVert_{ centerVert },
        fanData_{ fanData },
        points_{ points },
        normals_{ normals },
        useNeiNormals_{ useNeiNormals }
    {
        init_();
    }
    void optimize( int steps, float critAngle );
    void updateBorder( float angle = 0.9 * MR::PI_F );
private:
    Plane3f plane_;

    VertId centerVert_;
    TriangulatedFanData& fanData_;
    const VertCoords& points_;
    const VertCoords& normals_;
    bool useNeiNormals_ = true;

    void init_();

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
        res.stable = true;
        return res;
    }

    const auto& a = points_[centerVert_];
    const auto& b = points_[fanData_.neighbors[res.nextId]];
    const auto& c = points_[fanData_.neighbors[res.id]];
    const auto& d = points_[fanData_.neighbors[res.prevId]];

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

    // whether abc acd is allowed to be flipped to abd dbc
    bool flipPossibility = false;
    if ( useNeiNormals_ && ( dot( normals_[centerVert_], normals_[fanData_.neighbors[res.id]] ) < 0.0f ) )
        flipPossibility = true;
    else
        flipPossibility = isUnfoldQuadrangleConvex( a, b, c, d );

    if ( !( flipPossibility && ( deloneProf > 0.0f || angleProf > 0.0f ) ) )
    {
        res.stable = true;
        return res;
    }

    if ( deloneProf > 0.0f )
        res.weight += deloneProf;
    if ( angleProf > 0.0f )
        res.weight += angleProf;

    res.weight += planeDist / normVal;

    if ( useNeiNormals_ )
    {
        const auto cNorm = normals_[fanData_.neighbors[res.id]];
        res.weight += 5.0f * ( 1.0f - dot( normals_[centerVert_], cNorm ) );

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

    auto difAngle = fanData_.cacheAngleOrder[nextInd].first - fanData_.cacheAngleOrder[prevInd].first;
    if ( difAngle < 0.0 )
        difAngle += 2.0 * PI;

    constexpr double MIN_ANGLE = 0.05; // ~2 degrees

    if ( difAngle > MIN_ANGLE )
    {
        res.stable = true;
        return;
    }
    int otherId = nextEl ? res.nextId : res.prevId;
    auto lengthSq = ( points_[centerVert_] - points_[fanData_.neighbors[res.id]] ).lengthSq();
    auto otherLengthSq = ( points_[centerVert_] - points_[fanData_.neighbors[otherId]] ).lengthSq();
    if ( lengthSq < otherLengthSq )
    {
        res.stable = true;
        return;
    }

    res.weight = std::numeric_limits<float>::max();
}

void FanOptimizer::optimize( int steps, float critAng )
{
    updateBorder();
    if ( steps == 0 )
        return;

    std::priority_queue<FanOptimizerQueueElement> queue_;
    for ( int i = 0; i < fanData_.neighbors.size(); ++i )
        queue_.emplace( calcQueueElement_( i, critAng ) );

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
        queue_.emplace( calcQueueElement_( topEl.nextId, critAng ) );
        queue_.emplace( calcQueueElement_( topEl.prevId, critAng ) );
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
    const VertCoords& normals, float critAngle, bool useNeiNormals, int steps )
{
    if ( triangulationData.neighbors.empty() )
        return;
    FanOptimizer optimizer( points, normals, triangulationData, centerVert, useNeiNormals );
    optimizer.optimize( steps, critAngle );
}

void buildLocalTriangulation( const PointCloud& cloud, VertId v, const VertCoords& normals, const Settings & settings,
    TriangulatedFanData & fanData )
{
    findNeighbors( cloud, v, settings.radius, fanData.neighbors );
    if ( settings.useNeiNormals )
        filterNeighbors( normals, v, fanData.neighbors );
    trianglulateFan( cloud.points, v, fanData, normals, settings.critAngle, settings.useNeiNormals );

    float maxRadius = ( fanData.neighbors.size() < 2 ) ? settings.radius * 2 :
        updateNeighborsRadius( cloud.points, v, fanData.border, fanData.neighbors, settings.radius );

    if ( maxRadius > settings.radius )
    {
        // update triangulation if radius was increased
        findNeighbors( cloud, v, maxRadius, fanData.neighbors );
        if ( settings.useNeiNormals )
            filterNeighbors( normals, v, fanData.neighbors );
        trianglulateFan( cloud.points, v, fanData, normals, settings.critAngle, settings.useNeiNormals );
    }
}

bool isBoundaryPoint( const PointCloud& pointCloud, const VertCoords& normals,
    VertId v, float radius, float angle, TriangulatedFanData& triangulationData )
{
    TriangulationHelpers::findNeighbors( pointCloud, v, radius, triangulationData.neighbors );
    triangulationData.border = {};
    if ( triangulationData.neighbors.size() < 3 )
        return true;
    FanOptimizer optimizer( pointCloud.points, normals, triangulationData, v, true );
    optimizer.updateBorder( angle );
    return triangulationData.border.valid();
}

} //namespace TriangulationHelpers
} //namespace MR
