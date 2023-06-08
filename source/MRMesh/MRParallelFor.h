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

/// finds minimal and maximal elementsin vector in parallel
template<typename T>
std::pair<T, T> parallelMinMax( const std::vector<T>& vec )
{
    struct MinMax
    {
        T min;
        T max;
    };

    MinMax minElem{ vec[0], vec[0] };
    auto minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>( 1, vec.size() ), minElem,
    [&] ( const tbb::blocked_range<size_t> range, MinMax curMinMax )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            T val = vec[i];

            if ( val < curMinMax.min )
            {
                curMinMax.min = val;
            }
            if ( val > curMinMax.max )
            {
                curMinMax.max = val;
            }
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
