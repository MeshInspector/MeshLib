#include "MREPointCloudTriangulationHelpers.h"
#include "MRMesh/MRMeshDelone.h"
#include "MRMesh/MRVector3.h"
#include "MRMesh/MRPointCloud.h"
#include "MRMesh/MRPointsInBall.h"
#include "MRMesh/MRBestFit.h"
#include "MRMesh/MRPlane3.h"
#include <algorithm>

namespace
{
using namespace MR;

// check that edge angle is less then critical, and C point is further than B and D
bool checkTrisAngle( const Vector3f& a, const Vector3f& b, const Vector3f& c, const Vector3f& d, float critAng )
{
    auto ac = ( c - a );
    auto ab = ( b - a );
    auto ad = ( d - a );

    auto dirABC = cross( ab, ac );
    auto dirACD = cross( ac, ad );

    bool isGoodAngle = angle( dirABC, dirACD ) < critAng;
    if ( isGoodAngle )
        return true;

    bool isGoodLength = ac.lengthSq() < ( ( ab + ad ) * 0.5f ).lengthSq(); // not to make orphan C point (closest to A)
    return isGoodLength;
}
}

namespace MRE
{
using namespace MR;
namespace TriangulationHelpers
{

float updateNeighborsRadius( const VertCoords& points, VertId v, const std::vector<VertId>& fan, float baseRadius )
{
    float maxRadius = 0.0f;

    // increase radius if better local triangulation can exist
    for ( int i = 0; i < fan.size(); ++i )
        maxRadius = std::max( maxRadius, circumcircleDiameter( points[v],
                                                               points[fan[i]],
                                                               points[fan[( i + 1 ) % fan.size()]] ) );

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

TriangulatedFan trianglulateFan( const VertCoords& points, VertId centerVert, const std::vector<VertId>& neighbors, const Vector3f& norm,float critAngle, int steps /*= INT_MAX */ )
{
    TriangulatedFan res;
    if ( neighbors.size() < 2 )
        return {};

    // find best plane to sort points by projections angles
    PointAccumulator accum;
    accum.addPoint( Vector3d( points[centerVert] ) );
    for ( auto vid : neighbors )
        accum.addPoint( Vector3d( points[vid] ) );

    auto bestPlane = accum.getBestPlane();
    if ( dot( bestPlane.n, Vector3d( norm ) ) < 0.0 )
        bestPlane = -bestPlane; // invert plane, to have triplets with consistent normals

    Vector3d centerProj = bestPlane.project( Vector3d( points[centerVert] ) );
    Vector3d firstProj = bestPlane.project( Vector3d( points[neighbors[0]] ) );
    Vector3d baseVec = ( firstProj - centerProj ).normalized();

    // fill angles
    std::vector<std::pair<double, int>> angles( neighbors.size() );
    for ( int i = 0; i < neighbors.size(); ++i )
    {
        auto vec = ( bestPlane.project( Vector3d( points[neighbors[i]] ) ) - centerProj ).normalized();
        auto crossProd = cross( vec, baseVec );
        double sign = 1.0;
        if ( dot( crossProd, bestPlane.n ) < 0 )
            sign = -1.0;
        angles[i] = {std::atan2( sign * crossProd.length() ,dot( vec,baseVec ) ),i};
    }

    // sort candidates
    std::sort( angles.begin(), angles.end() );

    res.optimized.reserve( angles.size() );
    std::vector<VertId> edges( angles.size() );
    for ( int i = 0; i < angles.size(); ++i )
    {
        edges[i] = neighbors[angles[i].second];
        // check border fans
        auto diff = ( i + 1 < angles.size() ) ? ( angles[i + 1].first - angles[i].first ) : ( angles[0].first + 2.0 * PI - angles[i].first );
        if ( diff > PI )
            res.border = edges[i];
    }

    // optimize fan
    int allRemoves = 0;
    int removes = 0;
    do
    {
        removes = 0;
        res.optimized.clear();
        VertId pV = edges.back();
        VertId nV;

        for ( int i = 0; i < edges.size(); ++i )
        {
            VertId v = edges[i];
            if ( v == res.border || pV == res.border )
            {
                res.optimized.push_back( v );
                pV = v;
                continue;
            }

            if ( i == int( edges.size() ) - 1 && !res.optimized.empty() )
                nV = res.optimized.front();
            else
                nV = edges[( i + 1 ) % edges.size()];

            if ( allRemoves >= steps )
            {
                res.optimized.push_back( v );
                pV = v;
                continue;
            }

            if ( !checkDeloneQuadrangle( Vector3d( points[centerVert] ), Vector3d( points[nV] ), Vector3d( points[v] ), Vector3d( points[pV] ) ) ||
                 !checkTrisAngle( points[centerVert], points[nV], points[v], points[pV], critAngle ) )
            {
                ++removes;
                ++allRemoves;
            }
            else
            {
                res.optimized.push_back( v );
                pV = v;
            }
        }
        if ( allRemoves >= steps )
            break;
        edges = res.optimized;
    } while ( removes != 0 );

    return res;
}

}
}