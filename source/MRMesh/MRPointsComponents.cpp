#include "MRPointsComponents.h"
#include "MRPointCloud.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRProgressCallback.h"
#include "MRPch/MRTBB.h"
#include "MRBitSetParallelFor.h"
#include "MRPointCloudTriangulationHelpers.h"
#include "MRLocalTriangulations.h"


namespace MR
{

namespace PointCloudComponents
{

/// returns
/// 1. the mapping: VertId -> Root ID in [0, 1, 2, ...)
/// 2. the total number of roots
static std::pair<Vert2RegionMap, int> getUniqueRootIds( const VertMap& allRoots, const VertBitSet& region )
{
    MR_TIMER
    Vert2RegionMap uniqueRootsMap( allRoots.size() );
    int k = 0;
    for ( auto v : region )
    {
        auto& uniqIndex = uniqueRootsMap[allRoots[v]];
        if ( uniqIndex < 0 )
        {
            uniqIndex = RegionId( k );
            ++k;
        }
        uniqueRootsMap[v] = uniqIndex;
    }
    return { std::move( uniqueRootsMap ), k };
}

Expected<VertBitSet> getLargeComponentsUnion( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    assert( minSize > 1 );
    const auto& validPoints = pointCloud.validPoints;
    ProgressCallback subPc = subprogress( pc, 0.f, 0.9f );
    auto unionStructsRes = getUnionFindStructureVerts( pointCloud, maxDist, nullptr, subPc );
    if ( !unionStructsRes.has_value() )
        return unexpectedOperationCanceled();
    auto& unionStructs = *unionStructsRes;
    const auto& allRoots = unionStructs.roots();

    subPc = subprogress( pc, 0.9f, 0.95f );
    int counter = 0;
    const float counterMax = float( validPoints.count() );
    const int counterDivider = int( validPoints.count() ) / 100;
    HashMap<VertId, int> root2size;
    for ( auto v : validPoints )
    {
        ++root2size[allRoots[v]];
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    subPc = subprogress( pc, 0.95f, 1.f );
    counter = 0;
    VertBitSet result( validPoints.find_last() + 1 );
    for ( auto v : validPoints )
    {
        if ( root2size[allRoots[v]] >= minSize )
            result.set( v );
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    return result;
}

Expected<std::vector<VertBitSet>> getLargeComponents( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc /*= {} */ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    assert( minSize > 1 );
    const auto& validPoints = pointCloud.validPoints;
    ProgressCallback subPc = subprogress( pc, 0.f, 0.9f );
    auto unionStructsRes = getUnionFindStructureVerts( pointCloud, maxDist, nullptr, subPc );
    if ( !unionStructsRes.has_value() )
        return unexpectedOperationCanceled();
    auto& unionStructs = *unionStructsRes;
    const auto& allRoots = unionStructs.roots();

    subPc = subprogress( pc, 0.9f, 0.95f );
    int counter = 0;
    const float counterMax = float( validPoints.count() );
    const int counterDivider = int( validPoints.count() ) / 100;
    HashMap<VertId, int> root2size;
    for ( auto v : validPoints )
    {
        ++root2size[allRoots[v]];
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    subPc = subprogress( pc, 0.95f, 1.f );
    counter = 0;
    std::vector<VertBitSet> result;
    HashMap<VertId, size_t> root2index;
    const size_t validPointsSize = validPoints.find_last() + 1;
    for ( auto v : validPoints )
    {
        const VertId root = allRoots[v];
        if ( root2size[root] >= minSize )
        {
            auto [it, inserted] = root2index.insert( { root, result.size() } );
            if ( inserted )
            {
                result.push_back( VertBitSet( validPointsSize ) );
            }
            result[it->second].set( v );
        }
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    return result;
}

Expected<std::pair<std::vector<VertBitSet>, int>>  getAllComponents( const PointCloud& pointCloud, float maxDist,
    int maxComponentCount /*= INT_MAX*/, ProgressCallback pc /*= {} */ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    assert( maxComponentCount > 1 );
    const auto& validPoints = pointCloud.validPoints;
    ProgressCallback subPc = subprogress( pc, 0.f, 0.9f );
    auto unionStructsRes = getUnionFindStructureVerts( pointCloud, maxDist, nullptr, subPc );
    if ( !unionStructsRes.has_value() )
        return unexpectedOperationCanceled();
    auto& unionStructs = *unionStructsRes;
    const auto& allRoots = unionStructs.roots();

    subPc = subprogress( pc, 0.9f, 0.95f );
    auto [uniqueRootsMap, componentsCount] = getUniqueRootIds( allRoots, validPoints );
    if ( !componentsCount )
        return unexpected( std::string( "No components found." ) );

    const int componentsInGroup = maxComponentCount == INT_MAX ? 1 : ( componentsCount + maxComponentCount - 1 ) / maxComponentCount;
    if ( componentsInGroup != 1 )
        for ( RegionId& id : uniqueRootsMap )
            id = RegionId( id / componentsInGroup );
    componentsCount = ( componentsCount + componentsInGroup - 1 ) / componentsInGroup;
    std::vector<VertBitSet> res( componentsCount );

    // this block is needed to limit allocations for not packed meshes
    std::vector<int> resSizes( componentsCount, 0 );
    for ( auto v : validPoints )
    {
        int index = uniqueRootsMap[v];
        if ( v > resSizes[index] )
            resSizes[index] = v;
    }
    for ( int i = 0; i < componentsCount; ++i )
        res[i].resize( resSizes[i] + 1 );
    // end of allocation block

    for ( auto v : validPoints )
        res[uniqueRootsMap[v]].set( v );
    return std::pair<std::vector<VertBitSet>, int>{ res, componentsInGroup };
}

Expected<UnionFind<VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region /*= nullptr*/, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    const VertBitSet& vertsRegion = region ? *region : pointCloud.validPoints;

    if ( !vertsRegion.any() )
        return unexpected( std::string( "Chosen region empty" ) );

    const VertBitSet* lastPassVerts = &vertsRegion;
    const auto numVerts = vertsRegion.find_last() + 1;
    UnionFind<VertId> unionFindStructure( numVerts );
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    double timerRes[3];
    Timer timer( "getUnionFindStructureVerts 1" );
    ProgressCallback subPc = subprogress( pc, 0.0f, 0.3f );
#define LOCAL_TRIANG
#ifdef LOCAL_TRIANG
    auto optTriang = TriangulationHelpers::buildUnitedLocalTriangulations( pointCloud,
        { .radius = maxDist, .numNeis = 0, .automaticRadiusIncrease = false }, subPc );
#endif
    timer.finish();
    timerRes[0] = timer.secondsPassed().count();
#ifdef LOCAL_TRIANG
    if ( !optTriang )
        return unexpected( std::string( "Cannot make local triangulations" ) );
    const auto& triang = *optTriang;
    const auto& neighbors = triang.neighbors;
    const auto& fanRecords = triang.fanRecords;
#endif


    VertBitSet bdVerts;
    subPc = subprogress( pc, 0.3f, 1.0f );
    if ( numThreads > 1 )
    {
        bdVerts.resize( numVerts );
        lastPassVerts = &bdVerts;
        subPc = subprogress( pc, 0.3f, 0.7f );
        timer.start( "getUnionFindStructureVerts 2" );
        BitSetParallelForAllRanged( vertsRegion, [&] ( VertId v0, VertId, VertId vEnd )
        {
            if ( !contains( vertsRegion, v0 ) )
                return;

#ifdef LOCAL_TRIANG
            for ( uint32_t i = fanRecords[v0].firstNei; i < fanRecords[v0 + 1].firstNei; ++i )
            {
                const VertId& v1 = neighbors[i];
                if ( v0 < v1 && contains( vertsRegion, v1 ) )
                {
                    if ( v1 >= vEnd )
                        bdVerts.set( v0 );
                    else
                        unionFindStructure.unite( v0, v1 );
                }
            }
#else
            findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
                [&] ( VertId v1, const Vector3f& )
            {
                if ( v0 < v1 && contains( vertsRegion, v1 ) )
                {
                    if ( v1 >= vEnd )
                        bdVerts.set( v0 );
                    else
                        unionFindStructure.unite( v0, v1 );
                }
            } );
#endif
        }, subPc );
        timer.finish();
        timerRes[1] = timer.secondsPassed().count();
        if ( !reportProgress( subPc, 1.f ) )
            return unexpectedOperationCanceled();
        subPc = subprogress( pc, 0.7f, 1.f );
    }

    int counterProcessedVerts = 0;
    const float counterMax = float( lastPassVerts->count() );
    const int counterDivider = int( lastPassVerts->count() ) / 100;
    timer.start( "getUnionFindStructureVerts 3" );
    for ( auto v0 : *lastPassVerts )
    {
#ifdef LOCAL_TRIANG
        for ( uint32_t i = fanRecords[v0].firstNei; i < fanRecords[v0 + 1].firstNei; ++i )
        {
            const VertId& v1 = neighbors[i];
            if ( v0 < v1 && contains( vertsRegion, v1 ) )
            {
                unionFindStructure.unite( v0, v1 );
            }
        }
#else
        findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
            [&] ( VertId v1, const Vector3f& )
        {
            if ( v0 < v1 && contains( vertsRegion, v1 ) )
            {
                unionFindStructure.unite( v0, v1 );
            }
        } );
#endif
        ++counterProcessedVerts;
        if ( !reportProgress( subPc, counterProcessedVerts / counterMax, counterProcessedVerts, counterDivider ) )
            return unexpectedOperationCanceled();
    }
    timer.finish();
    timerRes[2] = timer.secondsPassed().count();
#ifdef LOCAL_TRIANG
    spdlog::info( "LOCAL TRIANG" );
#else
    spdlog::info( "POINTS IN BALL" );
#endif
    spdlog::info( "build ULT time = {} sec", timerRes[0] );
    spdlog::info( "unite parallel = {} sec", timerRes[1] );
    spdlog::info( "unite remnant = {} sec", timerRes[2]);

    return unionFindStructure;
}

}

}
