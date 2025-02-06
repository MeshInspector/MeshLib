#include "MROutliersPoints.h"
#include "MRPointCloud.h"
#include "MRProgressCallback.h"
#include "MRBitSetParallelFor.h"
#include "MRBestFit.h"
#include "MRPointsInBall.h"
#include "MRPointsComponents.h"

namespace MR
{

std::string FindOutliers::prepare( const PointCloud& pointCloud, float radius, OutlierTypeMask mask, ProgressCallback progress /*= {}*/ )
{
    validPoints_ = pointCloud.validPoints;

    if ( !validPoints_.any() )
        return "Empty Point Cloud.";

    if ( !( mask & OutlierType::All ) )
        return {};

    maskCached_ = mask;
    radius_ = radius;

    const bool calcSmallComponentsCached = mask & OutlierType::SmallComponents;
    const bool calcWeaklyConnectedCached = mask & OutlierType::WeaklyConnected;
    const bool calcFarSurfaceCached = mask & OutlierType::FarSurface;
    const bool calcBadNormalCached = mask & OutlierType::AwayNormal;


    const VertBitSet* lastPassVerts = &validPoints_;
    const auto numVerts = validPoints_.find_last() + 1;
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    if ( calcSmallComponentsCached )
        unionFindStructure_ = UnionFind<VertId>( numVerts );
    if ( calcWeaklyConnectedCached )
        weaklyConnectedStat_ = std::vector<uint8_t>( numVerts );
    if ( calcFarSurfaceCached )
        farSurfaceStat_ = std::vector<float>( numVerts );
    if ( calcBadNormalCached )
        badNormalStat_ = std::vector<float>( numVerts );

    VertBitSet secondPassVerts;
    ProgressCallback subProgress = subprogress( progress, 0.f, 0.4f );
    const auto& points = pointCloud.points;
    const auto& normals = pointCloud.normals;
    if ( numThreads > 1 )
    {
        secondPassVerts.resize( numVerts );
        lastPassVerts = &secondPassVerts;
        const bool continued = BitSetParallelForAllRanged( validPoints_, [&] ( VertId v0, const auto& range )
        {
            if ( !contains( validPoints_, v0 ) )
                return;
            int count = 0;
            PointAccumulator plane;
            Vector3f normalSum;
            findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], radius_,
                              [&] ( VertId v1, const Vector3f& )
            {
                if ( !contains( validPoints_, v1 ) )
                    return;
                if ( v1 != v0 )
                {
                    ++count;
                    if ( calcBadNormalCached )
                        normalSum += normals[v1];
                    if ( calcFarSurfaceCached )
                        plane.addPoint( points[v1] );
                }
                if ( calcSmallComponentsCached && v0 < v1 )
                {
                    if ( v1 >= range.end )
                        secondPassVerts.set( v0 );
                    else
                        unionFindStructure_.unite( v0, v1 );
                }
            } );
            if ( calcWeaklyConnectedCached )
                weaklyConnectedStat_[int( v0 )] = uint8_t( std::min( count, 255 ) );
            if ( calcFarSurfaceCached )
                farSurfaceStat_[int( v0 )] = count >= 3 ? plane.getBestPlanef().distance( points[v0] ) : 0.f;
            if ( calcBadNormalCached )
            {
                if ( normalSum.lengthSq() >= 0.09f )
                {
                    normalSum = normalSum.normalized();
                    badNormalStat_[int( v0 )] = angle( normals[v0], normalSum );
                }
                else
                    badNormalStat_[int( v0 )] = 180.f;
            }
        }, subProgress );

        if ( !continued )
            return stringOperationCanceled();
        subProgress = subprogress( progress, 0.4f, 1.f );
    }

    if ( calcSmallComponentsCached )
    {
        int counterProcessedVerts = 0;
        const int lastPassVertsCount = int( lastPassVerts->count() );
        const float counterMax = float( lastPassVertsCount );
        const int counterDivider = std::max( lastPassVertsCount / 100, 1 );
        for ( auto v0 : *lastPassVerts )
        {
            findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], radius_,
                              [&] ( VertId v1, const Vector3f& )
            {
                if ( v0 < v1 && contains( validPoints_, v1 ) )
                {
                    unionFindStructure_.unite( v0, v1 );
                }
            } );
            ++counterProcessedVerts;
            if ( !reportProgress( subProgress, counterProcessedVerts / counterMax, counterProcessedVerts, counterDivider ) )
                return stringOperationCanceled();
        }
    }

    return {};
}

void FindOutliers::setParams( const OutlierParams& params )
{
    params_ = params;
}

Expected<VertBitSet> FindOutliers::find( OutlierTypeMask mask, ProgressCallback progress /*= {}*/ )
{
    mask &= maskCached_;
    if ( !mask )
        return {};

    int maxTaskCount = 0;
    for ( int i = 0; i < 4; ++i )
        if ( ( 1 << i ) & mask )
            ++maxTaskCount;

    VertBitSet result;
    if ( mask & OutlierType::SmallComponents )
    {
        auto res = findSmallComponents( subprogress( progress, 0.f, 1.f / maxTaskCount ) );
        if ( !res.has_value() )
            return unexpected( res.error() );
        result |= *res;
    }
    if ( mask & OutlierType::WeaklyConnected )
    {
        auto res = findWeaklyConnected();
        if ( !res.has_value() )
            return unexpected( res.error() );
        result |= *res;
    }
    if ( mask & OutlierType::FarSurface )
    {
        auto res = findFarSurface();
        if ( !res.has_value() )
            return unexpected( res.error() );
        result |= *res;
    }
    if ( mask & OutlierType::AwayNormal )
    {
        auto res = findAwayNormal();
        if ( !res.has_value() )
            return unexpected( res.error() );
        result |= *res;
    }

    return result;
}

Expected<VertBitSet> FindOutliers::findSmallComponents( ProgressCallback progress /*= {}*/ )
{
    auto largeComponentsRes = PointCloudComponents::getLargeComponentsUnion( unionFindStructure_, validPoints_, params_.maxClusterSize + 1, progress );
    

    if ( !largeComponentsRes.has_value() )
        return unexpected( largeComponentsRes.error() );
    return validPoints_ - *largeComponentsRes;
}

Expected<VertBitSet> FindOutliers::findWeaklyConnected( ProgressCallback progress /*= {}*/ )
{
    VertBitSet result( validPoints_.find_last() + 1 );
    const bool continued = BitSetParallelFor( validPoints_, [&] ( VertId v )
    {
        if ( weaklyConnectedStat_[int( v )] <= params_.numNeigbors )
            result.set( v );
    }, progress );
    if ( !continued )
        return unexpectedOperationCanceled();
    return result;
}

Expected<VertBitSet> FindOutliers::findFarSurface( ProgressCallback progress /*= {}*/ )
{
    VertBitSet result( validPoints_.find_last() + 1 );
    const float realMinHeight = params_.minHeight * radius_;
    const bool continued = BitSetParallelFor( validPoints_, [&] ( VertId v )
    {
        if ( farSurfaceStat_[int( v )] > realMinHeight )
            result.set( v );
    }, progress );
    if ( !continued )
        return unexpectedOperationCanceled();
    return result;
}

Expected<VertBitSet> FindOutliers::findAwayNormal( ProgressCallback progress /*= {}*/ )
{
    VertBitSet result( validPoints_.find_last() + 1 );
    const bool continued = BitSetParallelFor( validPoints_, [&] ( VertId v )
    {
        if ( badNormalStat_[int( v )] > params_.minAngle )
            result.set( v );
    }, progress );
    if ( !continued )
        return unexpectedOperationCanceled();
    return result;
}

}
