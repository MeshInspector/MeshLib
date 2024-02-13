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

/// extended version BitSetParallelForAll with information about range
template <typename BS, typename F>
bool BitSetParallelForAllEx( const BS& bs, F f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    if ( !progressCb )
    {
        BitSetParallelForAll( bs, [&]( VertId v ){ f( v, VertId( 0 ) , VertId( bs.size() ) ); } );
        return true;
    }

    using IndexType = typename BS::IndexType;

    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    auto callingThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas( hardware_destructive_interference_size ) S
    {
        std::atomic<size_t> processedBits{ 0 };
    } s;
    static_assert( alignof( S ) == hardware_destructive_interference_size );
    static_assert( sizeof( S ) == hardware_destructive_interference_size );

    tbb::parallel_for( tbb::blocked_range<size_t>( 0, endBlock ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        const IndexType idBegin{ range.begin() * BS::bits_per_block };
        const IndexType idEnd{ range.end() < endBlock ? range.end() * BS::bits_per_block : bs.size() };
        size_t myProcessedBits = 0;
        const bool report = std::this_thread::get_id() == callingThreadId;
        for ( IndexType id = idBegin; id < idEnd; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( id, idBegin, idEnd );
            if ( ( ++myProcessedBits % reportProgressEveryBit ) == 0 )
            {
                if ( report )
                {
                    if ( !progressCb( float( myProcessedBits + s.processedBits.load( std::memory_order_relaxed ) ) / bs.size() ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
                else
                {
                    s.processedBits.fetch_add( myProcessedBits, std::memory_order_relaxed );
                    myProcessedBits = 0;
                }
            }
        }
        const auto total = s.processedBits.fetch_add( myProcessedBits, std::memory_order_relaxed );
        if ( report && !progressCb( float( total ) / bs.size() ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

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
        const int endBlock = int( bdVerts.size() + bdVerts.bits_per_block - 1 ) / bdVerts.bits_per_block;
        const auto bitsPerThread = ( endBlock + numThreads - 1 ) / numThreads * BitSet::bits_per_block;

        tbb::parallel_for( tbb::blocked_range<int>( 0, numThreads ),
            [&] ( const tbb::blocked_range<int>& range )
        {
            const VertId vBeg{ range.begin() * bitsPerThread };
            const VertId vEnd{ range.end() < numThreads ? range.end() * bitsPerThread : bdVerts.size() };
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
                            bdVerts.set( v0 );
                        else
                            unionFindStructure.unite( v0, v1 );
                    }
                } );
            }
        } );
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
