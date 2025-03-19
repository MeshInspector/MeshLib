#pragma once

#include "MRMeshFwd.h"

#include <atomic>
#include <thread>

namespace MR
{

/// helper class used to ensure that the specific thread is used only once
class ThreadSemaphore
{
public:
    MRMESH_API explicit ThreadSemaphore( std::thread::id id = std::this_thread::get_id() );

    /// RAII-style class to acquire and release the thread
    class Lock
    {
        friend class ThreadSemaphore;
        MRMESH_API Lock( ThreadSemaphore& semaphore );

    public:
        MRMESH_API ~Lock();

        /// check if the current thread is acquired successfully
        operator bool() const { return acquired_; }
        bool acquired() const { return acquired_; }

    private:
        std::thread::id id_;
        std::atomic<int>& semaphore_;
        bool acquired_{ false };
    };

    /// acquire the current thread
    MRMESH_API Lock acquire();

private:
    std::thread::id id_;
    std::atomic<int> semaphore_;
};

} // namespace MR
