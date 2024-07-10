#pragma once

#include "MRVector.h"
#include "MRPch/MRTBB.h"
#include "MRProgressCallback.h"
#include <atomic>
#include <limits>
#include <thread>

namespace MR
{

namespace Parallel
{

struct CallSimply
{
    auto operator() ( auto && f, auto id ) const { return f( id ); }
};

struct CallSimplyMaker
{
    auto operator() () const { return CallSimply{}; }
};

template<typename T>
struct CallWithTLS
{
    T & tls;
    auto operator() ( auto && f, auto id ) const { return f( id, tls ); }
};

template<typename L>
struct CallWithTLSMaker
{
    tbb::enumerable_thread_specific<L> & e;
    auto operator() () const { return CallWithTLS{ e.local() }; }
};

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
        const auto total = s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
        if ( report && !cb( float( total ) / size ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

} //namespace Parallel

/// \addtogroup BasicGroup
/// \{

/// executes given function f for each span element [begin, end)
template <typename I, typename F>
inline void ParallelFor( I begin, I end, F && f )
{
    Parallel::For( begin, end, Parallel::CallSimplyMaker{}, std::forward<F>( f ) );
}

/// executes given function f for each span element [begin, end), with periodic progress report
/// \return false if terminated by callback
template <typename I, typename F>
bool ParallelFor( I begin, I end, F && f, ProgressCallback cb, size_t reportProgressEvery = 1024 )
{
    return Parallel::For( begin, end, Parallel::CallSimplyMaker{}, std::forward<F>( f ), cb, reportProgressEvery );
}

/// executes given function f for each span element [begin, end)
/// passing e.local() (evaluated once for each sub-range) as the second argument to f
template <typename I, typename L, typename F>
inline void ParallelFor( I begin, I end, tbb::enumerable_thread_specific<L> & e, F && f )
{
    Parallel::For( begin, end, Parallel::CallWithTLSMaker{ e }, std::forward<F>( f ) );
}

/// executes given function f for each span element [begin, end), with periodic progress report
/// passing e.local() (evaluated once for each sub-range) as the second argument to f
/// \return false if terminated by callback
template <typename I, typename L, typename F>
inline bool ParallelFor( I begin, I end, tbb::enumerable_thread_specific<L> & e, F && f, ProgressCallback cb, size_t reportProgressEvery = 1024 )
{
    return Parallel::For( begin, end, Parallel::CallWithTLSMaker{ e }, std::forward<F>( f ), cb, reportProgressEvery );
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
