#include "MRPointsComponents.h"
#include "MRPointCloud.h"
#include "MRBitSet.h"
#include "MRTimer.h"
#include "MRPointsInBall.h"
#include "MRProgressCallback.h"
#include "MRPch/MRTBB.h"

namespace MR
{

namespace PointCloudComponents
{

Expected<MR::VertBitSet> getLargestComponentsUnion( const PointCloud& pointCloud, float maxDist, int minSize, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    assert( maxDist > 0.f );
    assert( minSize > 0 );
    const auto& validPoints = pointCloud.validPoints;
    ProgressCallback subPc = pc ? subprogress( pc, 0.f, 0.9f ) : pc;
    auto unionStructsRes = getUnionFindStructureVerts( pointCloud, maxDist, nullptr, subPc );
    if ( !unionStructsRes.has_value() )
        return unexpectedOperationCanceled();
    auto& unionStructs = *unionStructsRes;
    auto allRoots = unionStructs.roots();

    subPc = pc ? subprogress( pc, 0.9f, 0.95f ) : pc;
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

    subPc = pc ? subprogress( pc, 0.95f, 1.f ) : pc;
    counter = 0;
    VertBitSet result( validPoints.find_last() + 1 );
    for ( auto v : validPoints )
    {
        if ( root2size[allRoots[v]] > minSize )
            result.set( v );
        if ( !reportProgress( subPc, counter / counterMax, counter, counterDivider ) )
            return unexpectedOperationCanceled();
    }

    return result;
}

Expected<UnionFind<MR::VertId>> getUnionFindStructureVerts( const PointCloud& pointCloud, float maxDist, const VertBitSet* region /*= nullptr*/, ProgressCallback pc /*= {}*/ )
{
    MR_TIMER

    VertBitSet vertsRegion = pointCloud.validPoints;
    if ( region )
        vertsRegion &= *region;

    if ( !vertsRegion.any() )
        return unexpected( std::string( "Chosen region empty" ));

    const VertBitSet* lastPassVerts = &vertsRegion;
    const auto numVerts = vertsRegion.find_last() + 1;
    UnionFind<VertId> unionFindStructure( numVerts );
    const auto numThreads = int( tbb::global_control::active_value( tbb::global_control::max_allowed_parallelism ) );

    VertBitSet bdVerts;
    ProgressCallback subPc = pc ? subprogress( pc, 0.f, 1.0f ) : pc;
    if ( numThreads > 1 )
    {
        std::atomic<int> counterProcessedVerts = 0;
        std::atomic<bool> canceled = false;
        const float counterMax = float( vertsRegion.count() );
        const int counterDivider = int( vertsRegion.count() ) / 100;
        subPc = pc ? subprogress( pc, 0.f, 0.7f ) : pc;

        bdVerts.resize( numVerts );
        lastPassVerts = &bdVerts;
        const int endBlock = int( bdVerts.size() + bdVerts.bits_per_block - 1 ) / bdVerts.bits_per_block;
        const auto bitsPerThread = ( endBlock + numThreads - 1 ) / numThreads * BitSet::bits_per_block;

        tbb::parallel_for( tbb::blocked_range<int>( 0, numThreads ),
            [&] ( const tbb::blocked_range<int>& range )
        {
            if ( canceled )
                return;
            const VertId vBeg{ range.begin() * bitsPerThread };
            const VertId vEnd{ range.end() < numThreads ? range.end() * bitsPerThread : bdVerts.size() };
            int counter = 0;
            for ( auto v0 = vBeg; v0 < vEnd; ++v0 )
            {
                if ( !contains( vertsRegion, v0 ) )
                    continue;

                findPointsInBall( pointCloud.getAABBTree(), pointCloud.points[v0], maxDist,
                    [&] ( VertId v1, const Vector3f& )
                {
                    if ( v0 < v1 && contains( vertsRegion, v1 ) )
                    {
                        if ( v1 >= vEnd )
                            bdVerts.set( v0 ); // remember vert to unite later
                        else
                            unionFindStructure.unite( v0, v1 );
                    }
                } );
                ++counter;
                if ( pc && counter == counterDivider )
                {
                    if ( canceled )
                        return;
                    counterProcessedVerts += counter;
                    counter = 0;
                    if ( !subPc( counterProcessedVerts / counterMax ) )
                    {
                        canceled = true;
                        return;
                    }
                }
            }
            counterProcessedVerts += counter;
            if ( !reportProgress( subPc, counterProcessedVerts / counterMax, counter, counterDivider ) )
                canceled = true;
        } );
        if ( canceled )
            return unexpectedOperationCanceled();

        subPc = pc ? subprogress( pc, 0.7f, 1.f ) : pc;
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
