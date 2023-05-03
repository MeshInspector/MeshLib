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

/// \}

} // namespace MR
