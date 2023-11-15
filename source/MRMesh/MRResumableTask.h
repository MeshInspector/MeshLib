#pragma once

namespace MR
{

/// primitive coroutine-like task interface
template <typename T>
class ResumableTask
{
public:
    using result_type = T;

    virtual ~ResumableTask() = default;
    /// start the task
    virtual void start() = 0;
    /// resume the task, return true if the task is finished, false if it should be re-invoked later
    virtual bool resume() = 0;
    /// get the result
    virtual result_type result() const = 0;
};

} // namespace MR
