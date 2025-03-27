#pragma once

#include "MRPch/MRTBB.h"

namespace MR
{

/// allows several threads to work on a group of tasks in isolation (in one arena):
/// they cannot steal outside tasks until the all these tasks are finished;
/// this solves the issue of recursive calling of the function where TbbTaskArenaAndGroup was created
struct TbbTaskArenaAndGroup
{
    tbb::task_arena arena;
    tbb::task_group group;

    /// runs given function within this task group and task arena
    template<typename F>
    void execute( F&& f )
    {
        arena.execute( [this, f]()
        {
            group.run( [f]
            {
                f();
            } );
        } );
    }

    /// waits till all tasks are done, joining to their execution;
    /// wait() must be called before this class destruction
    void wait()
    {
        arena.execute( [this]
        {
            group.wait();
        } );
    }
};

} //namespace MR
