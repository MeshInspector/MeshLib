#include "MRParallelFor.h"

namespace MR
{

void parallelFor( size_t begin, size_t end, FunctionRef<void ( size_t )> f )
{
    tbb::parallel_for( tbb::blocked_range( begin, end ), [&f] ( const tbb::blocked_range<size_t>& range )
    {
        for ( auto i = range.begin(); i != range.end(); ++i )
            f( i );
    } );
}

void parallelFor( size_t begin, size_t end, FunctionRef<void ( size_t, void* )> f, FunctionRef<void* ()> ctx )
{
    tbb::parallel_for( tbb::blocked_range( begin, end ), [&f, &ctx] ( const tbb::blocked_range<size_t>& range )
    {
        void* ctx_ = ctx();
        for ( auto i = range.begin(); i != range.end(); ++i )
            f( i, ctx_ );
    } );
}

bool parallelFor( size_t begin, size_t end, FunctionRef<void ( size_t, void* )> f, FunctionRef<void* ()> ctx,
    ProgressCallback cb, size_t reportProgressEvery )
{
    if ( !cb )
    {
        parallelFor( begin, end, f, ctx );
        return true;
    }

    const auto size = end - begin;
    if ( size <= 0 )
        return true;

    TbbThreadMutex callingThreadMutex;
    std::atomic<bool> keepGoing{ true };

    // avoid false sharing with other local variables
    // by putting processedBits in its own cache line
    constexpr int hardware_destructive_interference_size = 64;
    struct alignas(hardware_destructive_interference_size) S
    {
        std::atomic<size_t> processed{ 0 };
    } s;
    static_assert( alignof(S) == hardware_destructive_interference_size );
    static_assert( sizeof(S) == hardware_destructive_interference_size );

    tbb::parallel_for( tbb::blocked_range( begin, end ), [&] ( const tbb::blocked_range<size_t>& range )
    {
        const auto callingThreadLock = callingThreadMutex.tryLock();
        const bool report = cb && callingThreadLock;
        size_t myProcessed = 0;

        void* ctx_ = ctx();
        for ( auto i = range.begin(); i < range.end(); ++i )
        {
            if ( !keepGoing.load( std::memory_order_relaxed ) )
                break;

            f( i, ctx_ );

            if ( ( ++myProcessed % reportProgressEvery ) == 0 )
            {
                if ( report )
                {
                    if ( !cb( float( myProcessed + s.processed.load( std::memory_order_relaxed ) ) / float( size ) ) )
                        keepGoing.store( false, std::memory_order_relaxed );
                }
                else
                {
                    s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
                    myProcessed = 0;
                }
            }
        }
        const auto total = myProcessed + s.processed.fetch_add( myProcessed, std::memory_order_relaxed );
        if ( report && !cb( float( total ) / float( size ) ) )
            keepGoing.store( false, std::memory_order_relaxed );
    } );
    return keepGoing.load( std::memory_order_relaxed );
}

} // namespace MR
