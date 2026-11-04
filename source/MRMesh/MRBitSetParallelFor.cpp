#include "MRBitSetParallelFor.h"

#include "MRTbbThreadMutex.h"
#include <bit>

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

// calls body on the subranges of blockRange in parallel threads, or on the whole range in this thread if it is one block,
// which a parallel loop would process in one thread anyway, only slower
// (unlike tbb::parallel_for, the one-block call neither binds to an enclosing task group nor cancels tgc on an exception)
template <typename Body, typename ...Tgc>
void forBlocks( const Range & blockRange, const Body & body, Tgc & ... tgc )
{
    if ( blockRange.size() == 1 )
        body( blockRange );
    else
        tbb::parallel_for( blockRange, body, tgc... );
}

// calls f( i ) for every set bit i of the blocks of subRange within bitSubRange, while f returns true; returns false if f stopped it
template <typename F>
bool forSetBits( const BitSet::Blocks & blocks, const Range & subRange, const Range & bitSubRange, F && f )
{
    for ( auto b = subRange.begin(); b < subRange.end(); ++b )
    {
        for ( auto bits = blocks[b]; bits; bits &= bits - 1 )
        {
            const auto i = b * BitSet::bits_per_block + std::countr_zero( bits );
            if ( i >= bitSubRange.end() )
                return true;
            if ( !f( i ) )
                return false;
        }
    }
    return true;
}

} // namespace

void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range & )> f )
{
    const auto blockRange = toBlockRange( bitRange );
    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
            f( i, bitSubRange );
    } );
}

void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range & )> f, tbb::task_group_context & tgc )
{
    const auto blockRange = toBlockRange( bitRange );
    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
        {
            if ( tgc.is_group_execution_cancelled() )
                break;
            f( i, bitSubRange );
        }
    }, tgc );
}

void forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx )
{
    const auto blockRange = toBlockRange( bitRange );
    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        void* ctx_ = ctx();
        for ( auto i = bitSubRange.begin(); i < bitSubRange.end(); ++i )
            f( i, bitSubRange, ctx_ );
    } );
}

void forSetRanged( const BitSet & bs, FunctionRef<void ( size_t, const Range & )> f )
{
    const Range bitRange( 0, bs.size() );
    const auto blockRange = toBlockRange( bitRange );
    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        forSetBits( bs.bits(), subRange, bitSubRange, [&] ( size_t i ) { f( i, bitSubRange ); return true; } );
    } );
}

void forSetRanged( const BitSet & bs, FunctionRef<void ( size_t, const Range & )> f, tbb::task_group_context & tgc )
{
    const Range bitRange( 0, bs.size() );
    const auto blockRange = toBlockRange( bitRange );
    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        forSetBits( bs.bits(), subRange, bitSubRange, [&] ( size_t i )
        {
            if ( tgc.is_group_execution_cancelled() )
                return false;
            f( i, bitSubRange );
            return true;
        } );
    }, tgc );
}

bool forSetRanged( const BitSet & bs, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx, const ProgressCallback& progressCb, size_t reportProgressEveryBit )
{
    const Range bitRange( 0, bs.size() );
    const auto blockRange = toBlockRange( bitRange );
    const auto & blocks = bs.bits();
    if ( !progressCb )
    {
        forBlocks( blockRange, [&] ( const Range & subRange )
        {
            const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
            void* ctx_ = ctx();
            forSetBits( blocks, subRange, bitSubRange, [&] ( size_t i ) { f( i, bitSubRange, ctx_ ); return true; } );
        } );
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

    forBlocks( blockRange, [&] ( const Range & subRange )
    {
        size_t myProcessedBits = 0, notReportedBits = 0;
        const auto callingThreadLock = callingThreadMutex.tryLock();

        const auto bitSubRange = toBitSubRange( bitRange, blockRange, subRange );
        void* ctx_ = ctx();
        for ( auto b = subRange.begin(); b < subRange.end(); ++b )
        {
            const bool completed = forSetBits( blocks, { b, b + 1 }, bitSubRange, [&] ( size_t i )
            {
                if ( !keepGoing.load( std::memory_order_relaxed ) )
                    return false;
                f( i, bitSubRange, ctx_ );
                return true;
            } );
            if ( !completed )
                break;

            // the progress counts all bits of the block, set or not
            const auto blockBits = std::min( ( b + 1 ) * BitSet::bits_per_block, bitSubRange.end() ) - std::max( b * BitSet::bits_per_block, bitSubRange.begin() );
            myProcessedBits += blockBits;
            notReportedBits += blockBits;
            if ( notReportedBits >= reportProgressEveryBit )
            {
                notReportedBits = 0;
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

bool forAllRanged( const Range & bitRange, FunctionRef<void ( size_t, const Range &, void* )> f,
    FunctionRef<void* ()> ctx, const ProgressCallback& progressCb, size_t reportProgressEveryBit )
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
    forBlocks( blockRange, [&] ( const Range & subRange )
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
