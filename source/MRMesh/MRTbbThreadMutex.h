#pragma once

#include "MRMeshFwd.h"

#include <atomic>
#include <thread>

namespace MR
{

/// helper class used to ensure that the specific thread is used only once
/// use it if you have nested TBB operations (e.g. parallel_for inside another parallel_for)
class TbbThreadMutex
{
public:
    MRMESH_API explicit TbbThreadMutex( std::thread::id id = std::this_thread::get_id() );

    /// ...
    class LockGuard
    {
        friend class TbbThreadMutex;
        TbbThreadMutex& mutex_;

        MRMESH_API explicit LockGuard( TbbThreadMutex& mutex );

    public:
        MRMESH_API ~LockGuard();
    };

    MRMESH_API std::optional<LockGuard> tryLock();

private:
    std::thread::id id_;
    std::atomic_flag lockFlag_;
};

} // namespace MR
