#pragma once

#include "MRBitSet.h"
#include "MRPch/MRTBB.h"
#include "MRProgressCallback.h"
#include <atomic>
#include <thread>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAll( const BS & bs, F f )
{
    using IndexType = typename BS::IndexType;

    const int endBlock = int( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    tbb::parallel_for( tbb::blocked_range<int>( 0, endBlock ), 
        [&]( const tbb::blocked_range<int> & range )
        {
            IndexType id{ range.begin() * BitSet::bits_per_block };
            const IndexType idEnd{ range.end() < endBlock ? range.end() * BS::bits_per_block : bs.size() };
            for ( ; id < idEnd; ++id )
            {
                f( id );
            }
        } );
}

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
/// uses tbb::static_partitioner for uniform distribution of ids;
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F>
bool BitSetParallelForAll( const BS& bs, F f, ProgressCallback progressCb )
{
    if ( !progressCb )
    {
        BitSetParallelForAll( bs, f );
        return true;
    }

    using IndexType = typename BS::IndexType;

    const int endBlock = int( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    tbb::parallel_for( tbb::blocked_range<int>( 0, endBlock ),
        [&] ( const tbb::blocked_range<int>& range )
    {
        IndexType id{ range.begin() * BitSet::bits_per_block };
        const IndexType idEnd{ range.end() < endBlock ? range.end() * BS::bits_per_block : bs.size() };
        auto idBegin = int( id );
        auto idRange = float( int( idEnd ) - idBegin );
        for ( ; id < idEnd; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( id );
            if ( std::this_thread::get_id() == mainThreadId )
            {
                if ( !progressCb( float( int( id ) - idBegin ) / idRange ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }
        }
    }, tbb::static_partitioner() ); // static partitioner is needed to uniform distribution of ids
    return keepGoing.load( std::memory_order_relaxed );
}

/// executes given function f for every set bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelFor( const BS& bs, F f )
{
    using IndexType = typename BS::IndexType;
    BitSetParallelForAll( bs, [&]( IndexType id )
    {
        if ( bs.test( id ) )
        {
            f( id );
        }
    } );
}

/// executes given function f for every set bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
/// uses tbb::static_partitioner for uniform distribution of ids
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F>
bool BitSetParallelFor( const BS& bs, F f, ProgressCallback progressCb )
{
    using IndexType = typename BS::IndexType;
    return BitSetParallelForAll( bs, [&] ( IndexType id )
    {
        if ( bs.test( id ) )
        {
            f( id );
        }
    }, progressCb );
}

/// \}

} // namespace MR
