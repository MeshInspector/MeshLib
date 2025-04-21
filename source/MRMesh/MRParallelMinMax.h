#pragma once

#include "MRParallelFor.h"
#include "MRMinMaxArg.h"
#include "MRBitSet.h"

namespace MR
{

/// finds minimal and maximal elements and their indices in given range [data, data+size) in parallel;
/// \param region if provided, only range values with indices corresponding to set bits here will be checked;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
MinMaxArg<T, size_t> parallelMinMaxArg( const T* data, size_t size, const BitSet* region = nullptr, const T* topExcluding = nullptr )
{
    using I = size_t;
    return tbb::parallel_reduce( tbb::blocked_range<I>( I(0), size ), MinMaxArg<T, I>{},
    [&] ( const tbb::blocked_range<I> range, MinMaxArg<T, I> curr )
    {
        for ( I i = range.begin(); i < range.end(); i++ )
        {
            if ( region && !region->test( i ) )
                continue;
            T val = data[i];
            if ( topExcluding )
            {
                T absVal = val;
                if constexpr ( !std::is_unsigned_v<T> )
                    absVal = (T)std::abs( val );
                if ( absVal >= *topExcluding )
                    continue;
            }
            curr.include( val, i );
        }
        return curr;
    },
    [&] ( MinMaxArg<T, I> a, const MinMaxArg<T, I>& b )
    {
        a.include( b );
        return a;
    } );
}

/// finds minimal and maximal elements and their indices in given vector in parallel;
/// \param region if provided, only vector values with indices corresponding to set bits here will be checked;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T, typename Itag>
MinMaxArg<T, Id<Itag>> parallelMinMaxArg( const Vector<T, Id<Itag>>& vec, const TaggedBitSet<Itag>* region = nullptr, const T* topExcluding = nullptr )
{
    auto mma = parallelMinMaxArg( vec.data(), vec.size(), region, topExcluding );
    return
    {
        .min = mma.min,
        .max = mma.max,
        .minArg = Id<Itag>( mma.minArg ),
        .maxArg = Id<Itag>( mma.maxArg )
    };
}

/// finds minimal and maximal elements in given range [data, data+size) in parallel;
/// \param region if provided, only range values with indices corresponding to set bits here will be checked;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
std::pair<T, T> parallelMinMax( const T* data, size_t size, const BitSet* region = nullptr, const T * topExcluding = nullptr )
{
    auto mma = parallelMinMaxArg( data, size, region, topExcluding );
    return { mma.min, mma.max };
}

/// finds minimal and maximal elements in given vector in parallel;
/// \param region if provided, only vector values with indices corresponding to set bits here will be checked;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T>
std::pair<T, T> parallelMinMax( const std::vector<T>& vec, const BitSet* region = nullptr, const T * topExcluding = nullptr )
{
    return parallelMinMax( vec.data(), vec.size(), region, topExcluding );
}

/// finds minimal and maximal elements in given vector in parallel;
/// \param region if provided, only vector values with indices corresponding to set bits here will be checked;
/// \param topExcluding if provided then all values in the array equal or larger by absolute value than it will be ignored
template<typename T, typename Itag>
std::pair<T, T> parallelMinMax( const Vector<T, Id<Itag>>& vec, const TaggedBitSet<Itag>* region = nullptr, const T* topExcluding = nullptr )
{
    return parallelMinMax( vec.data(), vec.size(), region, topExcluding );
}

} //namespace MR
