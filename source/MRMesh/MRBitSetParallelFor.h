#pragma once

#include "MRBitSet.h"
#include "MRParallel.h"
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
inline auto blockRange( const IdRange<IndexType> & bitRange )
{
    const size_t beginBlock = bitRange.beg / BitSet::bits_per_block;
    const size_t endBlock = ( size_t( bitRange.end ) + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block;
    return tbb::blocked_range<size_t>( beginBlock, endBlock );
}

template <typename BS>
inline auto blockRange( const BS & bs )
{
    const size_t endBlock = ( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    return tbb::blocked_range<size_t>( 0, endBlock );
}

template <typename BS>
inline auto bitRange( const BS & bs )
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

template <typename IndexType, typename CM, typename F>
void ForAllRanged( const IdRange<IndexType> & bitRange, const CM & callMaker, F && f )
{
    const auto range = BitSetParallel::blockRange( bitRange );
    tbb::parallel_for( range, [&]( const tbb::blocked_range<size_t> & subRange )
    {
        auto c = callMaker();
        const auto bitSubRange = BitSetParallel::bitSubRange( bitRange, range, subRange );
        for ( auto id = bitSubRange.beg; id < bitSubRange.end; ++id )
            c( f, id, bitSubRange );
    } );
}

template <typename BS, typename CM, typename F>
inline void ForAllRanged( const BS & bs, const CM & callMaker, F && f )
{
    ForAllRanged( bitRange( bs ), callMaker, std::forward<F>( f ) );
}

template <typename IndexType, typename CM, typename F> 
bool ForAllRanged( const IdRange<IndexType> & bitRange, const CM & callMaker, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    if ( !progressCb )
    {
        ForAllRanged( bitRange, callMaker, std::forward<F>( f ) );
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
        auto c = callMaker();
        for ( auto id = bitSubRange.beg; id < bitSubRange.end; ++id )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            c( f, id, bitSubRange );
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

template <typename BS, typename CM, typename F>
inline bool ForAllRanged( const BS & bs, const CM & callMaker, F && f, ProgressCallback progressCb, size_t reportProgressEveryBit = 1024 )
{
    return ForAllRanged( bitRange( bs ), callMaker, std::forward<F>( f ), progressCb, reportProgressEveryBit );
}

} // namespace BitSetParallel

/// executes given function f( bit, subBitRange ) for each bit in bitRange in parallel threads,
/// where (subBitRange) are the bits that will be processed by the same thread;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename ...F>
inline auto BitSetParallelForAllRanged( const BS & bs, F &&... f )
{
    return BitSetParallel::ForAllRanged( bs, Parallel::CallSimplyMaker{}, std::forward<F>( f )... );
}

/// executes given function f( bit, subBitRange, tls ) for each bit in IdRange or BitSet (bs) in parallel threads,
/// where subBitRange are the bits that will be processed by the same thread,
///       tls=e.local() (evaluated once for each subBitRange);
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename ...F>
inline auto BitSetParallelForAllRanged( const BS & bs, tbb::enumerable_thread_specific<L> & e, F &&... f )
{
    return BitSetParallel::ForAllRanged( bs, Parallel::CallWithTLSMaker<L>{ e }, std::forward<F>( f )... );
}

/// executes given function f for each index in IdRange or BitSet (bs) in parallel threads;
/// it is guaranteed that every individual block in BitSet is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename F, typename ...Cb>
inline auto BitSetParallelForAll( const BS & bs, F && f, Cb&&... cb )
{
    return BitSetParallel::ForAllRanged( bs, Parallel::CallSimplyMaker{}, [&f]( auto bit, auto && ) { f( bit ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for each index in IdRange or BitSet (bs) in parallel threads
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// it is guaranteed that every individual block in BitSet is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename F, typename ...Cb>
inline auto BitSetParallelForAll( const BS & bs, tbb::enumerable_thread_specific<L> & e, F && f, Cb&&... cb )
{
    return BitSetParallel::ForAllRanged( bs, Parallel::CallWithTLSMaker<L>{ e }, [&f]( auto bit, auto &&, auto & tls ) { f( bit, tls ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for every _set_ bit in IdRange or BitSet (bs) in parallel threads;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename F, typename ...Cb>
inline auto BitSetParallelFor( const BS& bs, F && f, Cb&&... cb )
{
    return BitSetParallelForAll( bs, [&]( auto bit ) { if ( bs.test( bit ) ) f( bit ); }, std::forward<Cb>( cb )... );
}

/// executes given function f for every _set_ bit in bs IdRange or BitSet (bs) parallel threads,
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// it is guaranteed that every individual block in bit-set is processed by one thread only;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEveryBit = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename BS, typename L, typename F, typename ...Cb>
inline auto BitSetParallelFor( const BS& bs, tbb::enumerable_thread_specific<L> & e, F && f, Cb&&... cb )
{
    return BitSetParallelForAll( bs, e, [&]( auto bit, auto & tls ) { if ( bs.test( bit ) ) f( bit, tls ); }, std::forward<Cb>( cb )... );
}

/// \}

} // namespace MR
