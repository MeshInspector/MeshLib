#include "MRPointsComponents.h"
#include "MRPointCloud.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRProgressCallback.h"
#include "MRPch/MRTBB.h"
#include "MRBitSetParallelFor.h"

namespace MR
{

namespace PointCloudComponents
{

Expected<MR::VertBitSet> getLargestComponentsUnion( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc /*= {}*/ )
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

Expected<UnionFind<MR::VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region /*= nullptr*/, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    const VertBitSet& vertsRegion = region ? *region : pointCloud.validPoints;

    if ( !vertsRegion.any() )
        return unexpected( std::string( "Chosen region empty" ));

    const VertBitSet* lastPassVerts = &vertsRegion;
    const auto numVerts = vertsRegion.find_last() + 1;
    UnionFind<VertId> unionFindStructure( numVerts );
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    VertBitSet bdVerts;
    ProgressCallback subPc = subprogress( pc, 0.f, 1.0f );
    if ( numThreads > 1 )
    {
        bdVerts.resize( numVerts );
        lastPassVerts = &bdVerts;
        subPc = subprogress( pc, 0.f, 0.7f );
        BitSetParallelForAllRanged( vertsRegion, [&] ( VertId v0, VertId, VertId vEnd )
        {
            if ( !contains( vertsRegion, v0 ) )
                return;
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
        }, subPc );
        if ( !reportProgress( subPc, 0.7f ) )
            return unexpectedOperationCanceled();
        subPc = subprogress( pc, 0.7f, 1.f );
    }

    int counterProcessedVerts = 0;
    const float counterMax = float( lastPassVerts->count() );
    const int counterDivider = int( lastPassVerts->count() ) / 100;
    for ( auto v0 : *lastPassVerts )
    {
        findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
            [&] ( VertId v1, const Vector3f& )
        {
            if ( v0 < v1 && contains( vertsRegion, v1 ) )
            {
                unionFindStructure.unite( v0, v1 );
            }
        } );
        ++counterProcessedVerts;
        if ( !reportProgress( subPc, counterProcessedVerts / counterMax, counterProcessedVerts, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    return unionFindStructure;
}

}

}
