#pragma once

#include "MRBitSet.h"
#include "MRPch/MRTBB.h"

namespace MR
{

// executes given function f for each bit in bs in parallel threads;
// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelForAll( const BS & bs, F f )
{
    using IndexType = typename BS::IndexType;

    const int endBlock = int( bs.size() + BS::bits_per_block - 1 ) / BS::bits_per_block;
    tbb::parallel_for( tbb::blocked_range<int>( 0, endBlock ), 
        [&]( const tbb::blocked_range<int> & range )
        {
            IndexType id{ range.begin() * BitSet::bits_per_block };
            const IndexType idEnd{ range.end() < endBlock ? range.end() * BS::bits_per_block : bs.size() };
            for ( ; id < idEnd; ++id )
            {
                f( id );
            }
        } );
}

// executes given function f for every set bit in bs in parallel threads;
// it is guaranteed that every individual block in bit-set is processed by one thread only
template <typename BS, typename F>
void BitSetParallelFor( const BS& bs, F f )
{
    using IndexType = typename BS::IndexType;
    BitSetParallelForAll( bs, [&]( IndexType id )
    {
        if ( bs.test( id ) )
        {
            f( id );
        }
    } );
}

} //namespace MR
