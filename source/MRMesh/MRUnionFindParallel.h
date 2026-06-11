#pragma once

#include "MRUnionFind.h"
#include "MRBitSetParallelFor.h"
#include "MRParallelFor.h"
#include "MRTimer.h"

namespace MR
{

/// constructs in parallel the bitset with 1-bits corresponding to root elements;
/// if region is provided then only its elements will be checked
template <typename I>
[[nodiscard]] TypedBitSet<I> findRootsBitSet( const BaseUnionFind<I> & uf, const TypedBitSet<I> * region = nullptr )
{
    MR_TIMER;
    TypedBitSet<I> res( uf.size() );
    BitSetParallelForAll( res, [&]( I i )
    {
        if ( region && !region->test( i ) )
            return;
        if ( uf.isRoot( i ) )
            res.set( i );
    } );
    return res;
}

/// constructs in parallel the bitset with 1-bits corresponding to the elements from same set as (a);
/// if region is provided then only its elements will be checked
template <typename I>
[[nodiscard]] TypedBitSet<I> findComponentBitSet( BaseUnionFind<I> & uf, I a, const TypedBitSet<I> * region = nullptr )
{
    MR_TIMER;
    TypedBitSet<I> res( uf.size() );
    a = uf.find( a );
    BitSetParallelForAllRanged( res, [&]( I i, const auto & range )
    {
        if ( region && !region->test( i ) )
            return;
        if ( a == uf.findUpdateRange( i, range.beg, range.end ) )
            res.set( i );
    } );
    return res;
}

/// returns true if there is no set in UnionFind that contains both an element from the given region and another element not from the region;
/// in other words, UnionFind contains a subdivision of both region and not-region on subsets
template <typename I>
[[nodiscard]] bool isSubdivision( const BaseUnionFind<I> & uf, const TypedBitSet<I> & region )
{
    MR_TIMER;
    tbb::task_group_context ctx;
    ParallelFor( I( 0 ), I( uf.size() ), [&]( I i )
    {
        if ( region.test( uf.parent( i ) ) != region.test( i ) )
            ctx.cancel_group_execution();
    }, ctx );
    return !ctx.is_group_execution_cancelled();
}

/// in parallel, makes the parent of each element point directly to its set root
/// (a multi-threaded BaseUnionFind::roots()); returns the updated parents vector
template <typename I>
const Vector<I, I> & updateRootsParallel( BaseUnionFind<I> & uf )
{
    MR_TIMER;
    // each thread compresses only its own contiguous range, so parent writes never collide;
    // the root walk reads may cross ranges, but every value seen leads to the same unchanging root
    tbb::parallel_for( tbb::blocked_range<I>( I( 0 ), I( uf.size() ) ),
        [&] ( const tbb::blocked_range<I> & range )
    {
        for ( I i = range.begin(); i < range.end(); ++i )
            uf.findUpdateRange( i, range.begin(), range.end() );
    } );
    return uf.parents();
}
} //namespace MR
