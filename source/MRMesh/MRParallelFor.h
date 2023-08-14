#pragma once

#include "MRVector.h"
#include "MRPch/MRTBB.h"
#include "MRProgressCallback.h"
#include <atomic>
#include <limits>
#include <thread>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

/// executes given function f for each span element [begin, end)
template <typename I, typename F>
void ParallelFor( I begin, I end, F && f )
{
    tbb::parallel_for( tbb::blocked_range( begin, end ),
        [&] ( const tbb::blocked_range<I>& range )
    {
        for ( I i = range.begin(); i < range.end(); ++i )
            f( i );
    } );
}

/// executes given function f for each span element [begin, end)
template <typename I, typename F>
bool ParallelFor( I begin, I end, F && f, ProgressCallback cb, size_t reportProgressEvery = 1024 )
{
    if ( !cb )
    {
        ParallelFor( begin, end, std::forward<F>( f ) );
        return true;
    }
    const auto size = end - begin;
    if ( size <= 0 )
        return true;

    auto callingThreadId = std::this_thread::get_id();
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
        const bool report = std::this_thread::get_id() == callingThreadId;
        size_t myProcessed = 0;
        for ( I i = range.begin(); i < range.end(); ++i )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;
            f( i );
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
        const auto total = s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
        if ( report && !cb( float( total ) / size ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

/// executes given function f for each vector element in parallel threads
template <typename T, typename ...F>
inline auto ParallelFor( const std::vector<T> & v, F &&... f )
{
    return ParallelFor( size_t(0), v.size(), std::forward<F>( f )... );
}

/// executes given function f for each vector element in parallel threads
template <typename T, typename I, typename ...F>
inline auto ParallelFor( const Vector<T, I> & v, F &&... f )
{
    return ParallelFor( v.beginId(), v.endId(), std::forward<F>( f )... );
}

/// finds minimal and maximal elements in given vector in parallel;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
std::pair<T, T> parallelMinMax( const std::vector<T>& vec, const T * topExcluding = nullptr )
{
    struct MinMax
    {
        T min = std::numeric_limits<T>::max();
        T max = std::numeric_limits<T>::lowest();
    };

    auto minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, vec.size() ), MinMax{},
    [&] ( const tbb::blocked_range<size_t> range, MinMax curMinMax )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            T val = vec[i];
            if ( topExcluding && std::abs( val ) >= *topExcluding )
                continue;
            if ( val < curMinMax.min )
                curMinMax.min = val;
            if ( val > curMinMax.max )
                curMinMax.max = val;
        }
        return curMinMax;
    },
    [&] ( const MinMax& a, const MinMax& b )
    {
        MinMax res;
        if ( a.min < b.min )
        {
            res.min = a.min;
        }
        else
        {
            res.min = b.min;
        }
        if ( a.max > b.max )
        {
            res.max = a.max;
        }
        else
        {
            res.max = b.max;
        }
        return res;
    } );

    return { minmax.min, minmax.max };
}

/// finds minimal and maximal elements and their indices in given vector in parallel;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T, typename I>
auto parallelMinMaxArg( const Vector<T, I>& vec, const T * topExcluding = nullptr )
{
    struct MinMaxArg
    {
        T min = std::numeric_limits<T>::max();
        T max = std::numeric_limits<T>::lowest();
        I minArg, maxArg;
    };

    return tbb::parallel_reduce( tbb::blocked_range<I>( I(0), vec.endId() ), MinMaxArg{},
    [&] ( const tbb::blocked_range<I> range, MinMaxArg curr )
    {
        for ( I i = range.begin(); i < range.end(); i++ )
        {
            T val = vec[i];
            if ( topExcluding && std::abs( val ) >= *topExcluding )
                continue;
            if ( val < curr.min )
            {
                curr.min = val;
                curr.minArg = i;
            }
            if ( val > curr.max )
            {
                curr.max = val;
                curr.maxArg = i;
            }
        }
        return curr;
    },
    [&] ( MinMaxArg a, const MinMaxArg& b )
    {
        if ( b.min < a.min )
        {
            a.min = b.min;
            a.minArg = b.minArg;
        }
        if ( b.max > a.max )
        {
            a.max = b.max;
            a.maxArg = b.maxArg;
        }
        return a;
    } );
}

/// \}

} // namespace MR
