#pragma once

#include "MRParallelFor.h"
#include "MRBox.h"
#include <limits>

namespace MR
{

/// finds minimal and maximal elements in given vector in parallel;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
std::pair<T, T> parallelMinMax( const T* data, size_t size, const T * topExcluding = nullptr )
{
    auto minmax = tbb::parallel_reduce( tbb::blocked_range<size_t>( 0, size ), MinMax<T>{},
    [&] ( const tbb::blocked_range<size_t> range, MinMax<T> curMinMax )
    {
        for ( size_t i = range.begin(); i < range.end(); i++ )
        {
            T val = data[i];
            if ( topExcluding )
            {
                T absVal = val;
                if constexpr ( !std::is_unsigned_v<T> )
                    absVal = (T)std::abs( val );
                if ( absVal >= *topExcluding )
                    continue;
            }
            if ( val < curMinMax.min )
                curMinMax.min = val;
            if ( val > curMinMax.max )
                curMinMax.max = val;
        }
        return curMinMax;
    },
    [&] ( const MinMax<T>& a, const MinMax<T>& b )
    {
        MinMax<T> res;
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

/// finds minimal and maximal elements in given vector in parallel;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
std::pair<T, T> parallelMinMax( const std::vector<T>& vec, const T * topExcluding = nullptr )
{
    return parallelMinMax( vec.data(), vec.size(), topExcluding );
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

} //namespace MR
