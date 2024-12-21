#pragma once
#include "MRMeshFwd.h"
#include <forward_list>
#include <atomic>
#include <thread>


namespace MR
{

/// This class allows progress to be reported from different threads.
/// Unlike progress callback that is passed to \ref ParallelFor, each task may report the progress separately,
/// and the progress displayed to user is not just a number of completed tasks divided by the total number of tasks,
/// but rather a (weighted) average of progresses reported from each task
class ParallelProgressReporter
{
private:
    struct TaskInfo
    {
        float progress = 0.f;
        float weight = 1.f;
    };

public:
    MRMESH_API ParallelProgressReporter( const ProgressCallback& cb );

    /// Local reporter. It should be passed as a callback to task.
    /// @note One local reporter must not be invoked concurrently.
    struct PerTaskReporter
    {
        MRMESH_API bool operator()( float p ) const;
        ParallelProgressReporter* reporter_ = nullptr;
        TaskInfo* task_ = nullptr;
    };


    /// Add task to the pull
    /// @note This function must not be invoked concurrently.
    /// @return The reporter functor, that could be safely invoked from a different thread
    MRMESH_API PerTaskReporter newTask( float weight = 1.f );

    /// Actually report the progress. Designed to be invoked in loop until all tasks are completed or until the operation is cancelled
    /// @note This function is automatically invoked from the \ref PerTaskReporter::operator() under the condition that the thread that invokes it
    ///     is the same as the thread from which ParallelProgressReporter was created.
    MRMESH_API bool operator()();

private:
    /// Invoked from local reporters concurrently.
    bool updateTask_( float delta );

    const ProgressCallback& cb_;

    /// progress of each task
    std::forward_list<TaskInfo> perTaskInfo_;

    /// sum of the weights of all the tasks
    std::atomic<float> totalWeight_;

    /// avg progress for all the tasks
    std::atomic<float> progress_ = 0;

    std::atomic<bool> continue_ = true;

    /// id if the thread from which the constructor was called (it is supposed to be the main thread)
    std::thread::id mainThreadId_;
};

}