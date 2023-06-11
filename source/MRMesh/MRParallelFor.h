#pragma once

#include "MRVector.h"
#include "MRPch/MRTBB.h"

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

/// executes given function f for each vector element in parallel threads
template <typename T, typename F>
void ParallelFor( const std::vector<T> & v, F && f )
{
    tbb::parallel_for( tbb::blocked_range( size_t(0), v.size() ),
        [&] ( const tbb::blocked_range<size_t>& range )
    {
        for ( size_t i = range.begin(); i < range.end(); ++i )
            f( i );
    } );
}

/// executes given function f for each vector element in parallel threads
template <typename T, typename I, typename F>
void ParallelFor( const Vector<T, I> & v, F && f )
{
    tbb::parallel_for( tbb::blocked_range( v.beginId(), v.endId() ),
        [&] ( const tbb::blocked_range<I>& range )
    {
        for ( I i = range.begin(); i < range.end(); ++i )
            f( i );
    } );
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


/// \}

} // namespace MR
