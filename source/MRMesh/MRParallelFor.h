#pragma once

#include "MRVector.h"
#include "MRProgressCallback.h"
#include "MRParallel.h"
#include "MRTbbThreadMutex.h"

#include <atomic>

namespace MR
{

namespace Parallel
{

template <typename I, typename CM, typename F>
void For( I begin, I end, const CM & callMaker, F && f )
{
    tbb::parallel_for( tbb::blocked_range( begin, end ),
        [&] ( const tbb::blocked_range<I>& range )
    {
        auto c = callMaker();
        for ( I i = range.begin(); i < range.end(); ++i )
            c( f, i );
    } );
}

template <typename I, typename CM, typename F>
bool For( I begin, I end, const CM & callMaker, F && f, ProgressCallback cb, size_t reportProgressEvery = 1024 )
{
    if ( !cb )
    {
        For( begin, end, callMaker, std::forward<F>( f ) );
        return true;
    }
    const auto size = end - begin;
    if ( size <= 0 )
        return true;

    TbbThreadMutex callingThreadMutex;
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas(hardware_destructive_interference_size) S
    {
        std::atomic<size_t> processed{ 0 };
    } s;
    static_assert( alignof(S) == hardware_destructive_interference_size );
    static_assert( sizeof(S) == hardware_destructive_interference_size );

    tbb::parallel_for( tbb::blocked_range( begin, end ),
        [&] ( const tbb::blocked_range<I>& range )
    {
        const auto callingThreadLock = callingThreadMutex.tryLock();
        const bool report = cb && callingThreadLock;
        size_t myProcessed = 0;
        auto c = callMaker();
        for ( I i = range.begin(); i < range.end(); ++i )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            c( f, i );
            if ( ( ++myProcessed % reportProgressEvery ) == 0 )
            {
                if ( report )
                {
                    if ( !cb( float( myProcessed + s.processed.load( std::memory_order_relaxed ) ) / size ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
                else
                {
                    s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
                    myProcessed = 0;
                }
            }
        }
        const auto total = myProcessed + s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
        if ( report && !cb( float( total ) / size ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

} //namespace Parallel

/// \addtogroup BasicGroup
/// \{

/// executes given function f for each span element [begin, end);
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename I, typename ...F>
inline auto ParallelFor( I begin, I end, F &&... f )
{
    return Parallel::For( begin, end, Parallel::CallSimplyMaker{}, std::forward<F>( f )... );
}

/// executes given function f for each span element [begin, end)
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename I, typename L, typename ...F>
inline auto ParallelFor( I begin, I end, tbb::enumerable_thread_specific<L> & e, F &&... f )
{
    return Parallel::For( begin, end, Parallel::CallWithTLSMaker<L>{ e }, std::forward<F>( f )... );
}

/// executes given function f for each vector element in parallel threads;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename T, typename ...F>
inline auto ParallelFor( const std::vector<T> & v, F &&... f )
{
    return ParallelFor( size_t(0), v.size(), std::forward<F>( f )... );
}

/// executes given function f for each vector element in parallel threads;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename T, typename I, typename ...F>
inline auto ParallelFor( const Vector<T, I> & v, F &&... f )
{
    return ParallelFor( v.beginId(), v.endId(), std::forward<F>( f )... );
}

/// \}

} // namespace MR
