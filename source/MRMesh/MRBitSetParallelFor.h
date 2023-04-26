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

/// executes given function f for each index in [begin, end) in parallel threads;
/// it is guaranteed that every individual block in BitSet is processed by one thread only
template <typename IndexType, typename F>
void BitSetParallelForAll( IndexType begin, IndexType end, F f )
{
    const size_t beginBlock = begin / BitSet::bits_per_block;
    const size_t endBlock = ( size_t( end ) + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, endBlock ), 
        [&]( const tbb::blocked_range<size_t> & range )
        {
            IndexType id{ range.begin() > beginBlock ? range.begin() * BitSet::bits_per_block : begin };
            const IndexType idEnd{ range.end() < endBlock ? range.end() * BitSet::bits_per_block : end };
            for ( ; id < idEnd; ++id )
            {
                f( id );
            }
        } );
}

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAll( const BS & bs, F f )
{
    using IndexType = typename BS::IndexType;

    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, endBlock ), 
        [&]( const tbb::blocked_range<size_t> & range )
        {
            IndexType id{ range.begin() * BS::bits_per_block };
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

    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    auto mainThreadId = std::this_thread::get_id();
    std::atomic<bool> keepGoing{ true };
    tbb::parallel_for( tbb::blocked_range<size_t>( 0, endBlock ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        IndexType id{ range.begin() * BS::bits_per_block };
        const IndexType idEnd{ range.end() < endBlock ? range.end() * BS::bits_per_block : bs.size() };
        auto idBegin = size_t( id );
        auto idRange = float( size_t( idEnd ) - idBegin );
        for ( ; id < idEnd; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( id );
            if ( std::this_thread::get_id() == mainThreadId )
            {
                if ( !progressCb( float( size_t( id ) - idBegin ) / idRange ) )
                    keepGoing.store( false, std::memory_order_relaxed );
            }
        }
    }, tbb::static_partitioner() ); // static partitioner is needed to uniform distribution of ids
    return keepGoing.load( std::memory_order_relaxed );
}

/// executes given function f for every set bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F, typename ...Cb>
auto BitSetParallelFor( const BS& bs, F f, Cb&&... cb )
{
    using IndexType = typename BS::IndexType;
    return BitSetParallelForAll( bs, [&]( IndexType id )
    {
        if ( bs.test( id ) )
        {
            f( id );
        }
    }, std::forward<Cb>( cb )... );
}

/// executes given function f for every reset bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F, typename ...Cb>
auto BitSetParallelForReset( const BS& bs, F f, Cb&&... cb )
{
    using IndexType = typename BS::IndexType;
    return BitSetParallelForAll( bs, [&]( IndexType id )
    {
        if ( !bs.test( id ) )
        {
            f( id );
        }
    }, std::forward<Cb>( cb )... );
}

/// \}

} // namespace MR
