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
    auto size() const { return end - beg; }
};

namespace BitSetParallel
{

template <typename IndexType>
auto blockRange( const IdRange<IndexType> & bitRange )
{
    const size_t beginBlock = bitRange.beg / BitSet::bits_per_block;
    const size_t endBlock = ( size_t( bitRange.end ) + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block;
    return tbb::blocked_range<size_t>( beginBlock, endBlock );
}

template <typename BS>
auto blockRange( const BS & bs )
{
    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    return tbb::blocked_range<size_t>( 0, endBlock );
}

template <typename BS>
auto bitRange( const BS & bs )
{
    return IdRange<typename BS::IndexType>{ bs.beginId(), bs.endId() };
}

template <typename IndexType>
auto bitSubRange( const IdRange<IndexType> & bitRange, const tbb::blocked_range<size_t> & range, const tbb::blocked_range<size_t> & subRange )
{
    return IdRange<IndexType>
    {
        .beg = subRange.begin() > range.begin() ? IndexType( subRange.begin() * BitSet::bits_per_block ) : bitRange.beg,
        .end = subRange.end() < range.end()     ? IndexType( subRange.end()   * BitSet::bits_per_block ) : bitRange.end
    };
}

} // namespace BitSetParallel

/// executes given function f( bit, subBitRange ) for each bit in bitRange in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename IndexType, typename F>
void BitSetParallelForAllRanged( const IdRange<IndexType> & bitRange, F && f )
{
    const auto range = BitSetParallel::blockRange( bitRange );
    tbb::parallel_for( range, [&]( const tbb::blocked_range<size_t> & subRange )
    {
        const auto bitSubRange = BitSetParallel::bitSubRange( bitRange, range, subRange );
        for ( auto id = bitSubRange.beg; id < bitSubRange.end; ++id )
            f( id, bitSubRange );
    } );
}

/// executes given function f for each index in bitRange in parallel threads;
/// it is guaranteed that every individual block in BitSet is processed by one thread only
template <typename IndexType, typename F>
void BitSetParallelForAll( const IdRange<IndexType> & bitRange, F && f )
{
    BitSetParallelForAllRanged( bitRange, [&f]( auto bit, auto && ) { f( bit ); } );
}

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAll( const BS & bs, F && f )
{
    BitSetParallelForAll( BitSetParallel::bitRange( bs ), std::forward<F>( f ) );
}

/// executes given function f( bit, subBitRange ) for each bit in bs in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAllRanged( const BS& bs, F && f )
{
    return BitSetParallelForAllRanged( BitSetParallel::bitRange( bs ), std::forward<F>( f ) );
}

/// executes given function f( bit, subBitRange ) for each bit in bitRange in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// reports progress from calling thread only;
/// \return false if the processing was canceled by progressCb
template <typename IndexType, typename F> 
bool BitSetParallelForAllRanged( const IdRange<IndexType> & bitRange, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    if ( !progressCb )
    {
        BitSetParallelForAllRanged( bitRange, std::forward<F>( f ) );
        return true;
    }

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

    const auto range = BitSetParallel::blockRange( bitRange );
    tbb::parallel_for( range, [&] ( const tbb::blocked_range<size_t>& subRange )
    {
        const auto bitSubRange = BitSetParallel::bitSubRange( bitRange, range, subRange );
        size_t myProcessedBits = 0;
        const bool report = std::this_thread::get_id() == callingThreadId;
        for ( auto id = bitSubRange.beg; id < bitSubRange.end; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( id, bitSubRange );
            if ( ( ++myProcessedBits % reportProgressEveryBit ) == 0 )
            {
                if ( report )
                {
                    if ( !progressCb( float( myProcessedBits + s.processedBits.load( std::memory_order_relaxed ) ) / bitRange.size() ) )
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
        if ( report && !progressCb( float( total ) / bitRange.size() ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

/// executes given function f( bit, subBitRange ) for each bit in bs in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// reports progress from calling thread only;
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F> 
bool BitSetParallelForAllRanged( const BS& bs, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    return BitSetParallelForAllRanged( BitSetParallel::bitRange( bs ), std::forward<F>( f ), progressCb, reportProgressEveryBit );
}

/// executes given function f for each bit in bs in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// reports progress from calling thread only;
/// \return false if the processing was canceled by progressCb
template <typename BS, typename F>
bool BitSetParallelForAll( const BS& bs, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    return BitSetParallelForAllRanged( BitSetParallel::bitRange( bs ), [&f]( auto bit, auto && ) { f( bit ); }, progressCb, reportProgressEveryBit );
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
