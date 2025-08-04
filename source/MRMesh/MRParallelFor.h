#pragma once

#include "MRFunctional.h"
#include "MRParallel.h"
#include "MRProgressCallback.h"
#include "MRTbbThreadMutex.h"
#include "MRVector.h"

#include <atomic>

namespace MR
{

/// \addtogroup BasicGroup
/// \{

MRMESH_API void parallelFor( size_t begin, size_t end, FunctionRef<void ( size_t )> f );

MRMESH_API void parallelFor( size_t begin, size_t end, FunctionRef<void ( void*, size_t )> f, FunctionRef<void* ()> ctx );

MRMESH_API bool parallelFor( size_t begin, size_t end, FunctionRef<void ( void*, size_t )> f, FunctionRef<void* ()> ctx,
    ProgressCallback cb, size_t reportProgressEvery = 1024 );

/// executes given function f for each span element [begin, end);
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename I, typename F, typename ...Args>
inline auto ParallelFor( I begin, I end, F && f, Args && ... args )
{
    if constexpr ( sizeof...( args ) == 0 )
    {
        return parallelFor( begin, end, [&] ( size_t i )
        {
            std::forward<F>( f )( I( i ) );
        } );
    }
    else
    {
        return parallelFor( begin, end, [&] ( void*, size_t i )
        {
            std::forward<F>( f )( I( i ) );
        }, [&]
        {
            return nullptr;
        }, std::forward<Args>( args )... );
    }
}

/// executes given function f for each span element [begin, end)
/// passing e.local() (evaluated once for each sub-range) as the second argument to f;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename I, typename L, typename F, typename ...Args>
inline auto ParallelFor( I begin, I end, tbb::enumerable_thread_specific<L> & e, F && f, Args && ... args )
{
    return parallelFor( begin, end, [&] ( void* ctx, size_t i )
    {
        std::forward<F>( f )( I( i ), *(L*)ctx );
    }, [&]
    {
        return &e.local();
    }, std::forward<Args>( args )... );
}

/// executes given function f for each vector element in parallel threads;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename T, typename F, typename ...Args>
inline auto ParallelFor( const std::vector<T> & v, F && f, Args && ... args )
{
    return ParallelFor( size_t(0), v.size(), std::forward<F>( f ), std::forward<Args>( args )... );
}

/// executes given function f for each vector element in parallel threads;
/// optional parameters after f: ProgressCallback cb, size_t reportProgressEvery = 1024 for periodic progress report
/// \return false if terminated by callback
template <typename T, typename I, typename F, typename ...Args>
inline auto ParallelFor( const Vector<T, I> & v, F && f, Args && ... args )
{
    return ParallelFor( v.beginId(), v.endId(), std::forward<F>( f ), std::forward<Args>( args )... );
}

/// \}

} // namespace MR
