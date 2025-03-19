#pragma once

#include "MRMeshFwd.h"

#include <atomic>
#include <optional>
#include <thread>

namespace MR
{

/// helper class used to ensure that the specific thread is not re-used by TBB
/// use it if you have nested TBB operations (e.g. parallel_for inside another parallel_for)
/// \code
/// TbbThreadMutex reportThreadMutex;
/// tbb::parallel_for( range, [&] ( auto&& range )
/// {
///     const auto reportThreadLock = reportThreadMutex.tryLock();
///     for ( auto i = range.begin(); i != range.end(); ++i )
///     {
///         // if you have a nested `parallel_for` call here, you might come back to the same main loop with another range
///         // since TBB can re-use 'stale' threads
///         tbb::parallel_for( ... );
///         if ( reportThreadLock )
///             report( ... );
///     }
/// } );
/// \endcode
class MR_BIND_IGNORE MRMESH_CLASS TbbThreadMutex
{
public:
    /// construct class
    /// \param id - id of thread allowed to lock the mutex
    MRMESH_API explicit TbbThreadMutex( std::thread::id id = std::this_thread::get_id() );

    /// RAII-style lock guard for the mutex; releases it on destruction
    class LockGuard
    {
        friend class TbbThreadMutex;
        TbbThreadMutex& mutex_;

        explicit LockGuard( TbbThreadMutex& mutex ) : mutex_( mutex ) {}

    public:
        ~LockGuard() { mutex_.lockFlag_.clear(); }
    };

    /// try to lock the mutex
    /// returns a lock guard if the current thread id is equal to the mutex's one and the mutex is not locked yet
    MRMESH_API std::optional<LockGuard> tryLock();

private:
    std::thread::id id_;
    std::atomic_flag lockFlag_;
};

} // namespace MR
