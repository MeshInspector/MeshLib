#pragma once

#include "MRUnionFind.h"
#include "MRBitSetParallelFor.h"
#include "MRTimer.h"

namespace MR
{

/// constructs in parallel the bitset with 1-bits corresponding to root elements;
/// if region is provided then only its elements will be checked
template <typename I>
TypedBitSet<I> findRootsBitSet( const UnionFind<I> & uf, const TypedBitSet<I> * region = nullptr )
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
TypedBitSet<I> findComponentBitSet( UnionFind<I> & uf, I a, const TypedBitSet<I> * region = nullptr )
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

} //namespace MR
