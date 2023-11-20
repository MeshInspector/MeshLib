#pragma once

#include <memory>
#include <optional>

namespace MR
{

/// primitive coroutine-like task interface
template <typename T>
class MRMESH_CLASS ResumableTask
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

/// helper class to post-process task result
template <typename T, typename U>
class PostProcessResumableTask : public ResumableTask<U>
{
public:
    PostProcessResumableTask( std::shared_ptr<ResumableTask<T>> task, std::function<U ( T&& )> postProcess )
        : task_( std::move( task ) )
        , postProcess_( std::move( postProcess ) )
    {}
    ~PostProcessResumableTask() override = default;

    void start() override
    {
        task_->start();
    }

    bool resume() override
    {
        if ( !task_->resume() )
            return false;

        if constexpr ( !std::is_void_v<U> )
            result_ = postProcess_( task_->result() );
        else
            postProcess_( task_->result() );
        return true;
    }

    [[nodiscard]] U result() const override
    {
        if constexpr ( !std::is_void_v<U> )
            return *result_;
    }

private:
    std::shared_ptr<ResumableTask<T>> task_;
    std::function<U ( T&& )> postProcess_;
    std::optional<std::conditional<std::is_void_v<U>, bool, U>> result_;
};

} // namespace MR
