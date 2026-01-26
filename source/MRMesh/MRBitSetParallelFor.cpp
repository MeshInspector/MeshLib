#include "MRBitSetParallelFor.h"

#include "MRTbbThreadMutex.h"

namespace MR::BitSetParallel
{

namespace
{

Range toBlockRange( const Range & bitRange )
{
    return {
        bitRange.begin() / BitSet::bits_per_block,
        ( bitRange.end() + BitSet::bits_per_block - 1 ) / BitSet::bits_per_block,
    };
}

Range toBitSubRange( const Range & bitRange, const Range & blockRange, const Range & subRange )
{
    return {
        subRange.begin() > blockRange.begin() ? subRange.begin() * BitSet::bits_per_block : bitRange.begin(),
        subRange.end() < blockRange.end() ? subRange.end() * BitSet::bits_per_block : bitRange.end(),
    };
}

} // namespace

void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range & )> f )
{
    const auto blockRange = toBlockRange( bitRange );
    tbb::parallel_for( blockRange, [&] ( const Range & subRange )
    {
        assert( subRange.begin() + 1 == subRange.end() );
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
            f( i, bitSubRange );
    } );
}

void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx )
{
    const auto blockRange = toBlockRange( bitRange );
    tbb::parallel_for( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        void* ctx_ = ctx();
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
            f( i, bitSubRange, ctx_ );
    } );
}

bool forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx, ProgressCallback progressCb, size_t reportProgressEveryBit )
{
    if ( !progressCb )
    {
        forAllRanged( bitRange, f, ctx );
        return true;
    }

    TbbThreadMutex callingThreadMutex;
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas( hardware_destructive_interference_size ) S
    {
        std::atomic<size_t> processedBits{ 0 };
    } s;
    static_assert( alignof( S ) == hardware_destructive_interference_size );
    static_assert( sizeof( S ) == hardware_destructive_interference_size );

    const auto blockRange = toBlockRange( bitRange );
    tbb::parallel_for( blockRange, [&] ( const Range & subRange )
    {
        size_t myProcessedBits = 0;
        const auto callingThreadLock = callingThreadMutex.tryLock();

        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        void* ctx_ = ctx();
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;

            f( i, bitSubRange, ctx_ );

            if ( ( ++myProcessedBits % reportProgressEveryBit ) == 0 )
            {
                if ( callingThreadLock )
                {
                    if ( !progressCb( float( myProcessedBits + s.processedBits.load( std::memory_order_relaxed ) ) / float( bitRange.size() ) ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
                else
                {
                    s.processedBits.fetch_add( myProcessedBits, std::memory_order_relaxed );
                    myProcessedBits = 0;
                }
            }
        }

        const auto total = myProcessedBits + s.processedBits.fetch_add( myProcessedBits, std::memory_order_relaxed );
        if ( callingThreadLock && !progressCb( float( total ) / float( bitRange.size() ) ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );

    return keepGoing.load( std::memory_order_relaxed );
}

} // namespace MR
