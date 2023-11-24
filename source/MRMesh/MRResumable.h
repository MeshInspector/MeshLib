#pragma once

#include <functional>
#include <optional>

namespace MR
{

/// function with delayed result
/// returns std::nullopt until the result value is ready
template <typename Result>
using Resumable = std::function<std::optional<Result> ()>;

/// preserve the function's result to avoid calling it twice
template <typename Result>
class CachedResumable
{
public:
    explicit CachedResumable( Resumable<Result> f )
        : f_( std::move( f ) )
    {
        //
    }

    std::optional<Result> operator ()()
    {
        if ( !result_ )
            result_ = f_();
        return result_;
    }

private:
    Resumable<Result> f_;
    std::optional<Result> result_;
};

template <typename Result>
auto cached( Resumable<Result> f )
{
    return CachedResumable( std::move( f ) );
}

} // namespace MR
