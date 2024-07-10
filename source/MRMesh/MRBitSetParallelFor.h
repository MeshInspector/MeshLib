#pragma once

#include "MRBitSet.h"
#include "MRPch/MRTBB.h"
#include "MRProgressCallback.h"
#include <atomic>
#include <cassert>
#include <thread>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// range of indices [beg, end)
template <typename Id>
struct IdRange
{
    Id beg, end;
};

namespace BitSetParallel
{

template <typename IndexType>
auto blockRange( IndexType begin, IndexType end )
{
    const size_t beginBlock = begin / BitSet::bits_per_block;
    const size_t endBlock = ( size_t( end ) + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block;
    return tbb::blocked_range<size_t>( beginBlock, endBlock );
}

template <typename BS>
auto blockRange( const BS & bs )
{
    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    return tbb::blocked_range<size_t>( 0, endBlock );
}

template <typename IndexType>
auto bitRange( IndexType begin, IndexType end, const tbb::blocked_range<size_t> & range, const tbb::blocked_range<size_t> & subrange )
{
    return IdRange<IndexType>
    {
        .beg = subrange.begin() > range.begin() ? IndexType( subrange.begin() * BitSet::bits_per_block ) : begin,
        .end = subrange.end() < range.end() ? IndexType( subrange.end() * BitSet::bits_per_block ) : end
    };
}

template <typename BS>
auto bitRange( const BS & bs, const tbb::blocked_range<size_t> & range, const tbb::blocked_range<size_t> & subrange )
{
    assert( range.begin() == 0 );
    using I = typename BS::IndexType;
    return IdRange<I>
    {
        .beg = I( subrange.begin() * BS::bits_per_block ),
        .end = I( subrange.end() < range.end() ? subrange.end() * BS::bits_per_block : bs.size() )
    };
}

} // namespace BitSetParallel

/// executes given function f for each index in [begin, end) in parallel threads;
/// it is guaranteed that every individual block in BitSet is processed by one thread only
template <typename IndexType, typename F>
void BitSetParallelForAll( IndexType begin, IndexType end, F && f )
{
    const auto range = BitSetParallel::blockRange( begin, end );
    tbb::parallel_for( range, [&]( const tbb::blocked_range<size_t> & subrange )
    {
        const auto bitRange = BitSetParallel::bitRange( begin, end, range, subrange );
        for ( auto id = bitRange.beg; id < bitRange.end; ++id )
            f( id );
    } );
}

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAll( const BS & bs, F && f )
{
    BitSetParallelForAll( bs.beginId(), bs.endId(), std::forward<F>( f ) );
}

/// executes given function f( bit, range ) for each bit in bs in parallel threads,
/// where (range) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAllRanged( const BS& bs, F && f )
{
    const auto range = BitSetParallel::blockRange( bs );
    tbb::parallel_for( range, [&] ( const tbb::blocked_range<size_t>& subrange )
    {
        const auto bitRange = BitSetParallel::bitRange( bs, range, subrange );
        for ( auto id = bitRange.beg; id < bitRange.end; ++id )
            f( id, bitRange );
    } );
}

/// executes given function f( bit, range ) for each bit in bs in parallel threads,
/// where (range) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// reports progress from calling thread only;
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F> 
bool BitSetParallelForAllRanged( const BS& bs, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    if ( !progressCb )
    {
        BitSetParallelForAllRanged( bs, std::forward<F>( f ) );
        return true;
    }

    const auto range = BitSetParallel::blockRange( bs );
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

    tbb::parallel_for( range, [&] ( const tbb::blocked_range<size_t>& subrange )
    {
        const auto bitRange = BitSetParallel::bitRange( bs, range, subrange );
        size_t myProcessedBits = 0;
        const bool report = std::this_thread::get_id() == callingThreadId;
        for ( auto id = bitRange.beg; id < bitRange.end; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( id, bitRange );
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

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// reports progress from calling thread only;
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F>
bool BitSetParallelForAll( const BS& bs, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    return BitSetParallelForAllRanged( bs, [&f]( auto bit, auto && ) { f( bit ); }, progressCb, reportProgressEveryBit );
}

/// executes given function f for every _set_ bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F, typename ...Cb>
auto BitSetParallelFor( const BS& bs, F && f, Cb&&... cb )
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

/// executes given function f for every _reset_ bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F, typename ...Cb>
auto BitSetParallelForReset( const BS& bs, F && f, Cb&&... cb )
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
