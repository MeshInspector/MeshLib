#pragma once

#include <functional>
#include <variant>

namespace MR::StateMachine
{

struct YieldT {};

template <typename State>
struct ContinueWithT
{
    State state;
};

template <typename Result>
struct FinishWithT
{
    Result result;
};

template <typename State, typename Result>
using Transition = std::variant<YieldT, ContinueWithT<State>, FinishWithT<Result>>;

template <typename State, typename Result>
using Action = std::function<Transition<State, Result> ( State )>;

constexpr YieldT Yield;

template <typename State>
ContinueWithT<State> ContinueWith( State&& state )
{
    return { std::forward<State>( state ) };
}

template <typename Result>
FinishWithT<Result> FinishWith( Result&& result )
{
    return { std::forward<Result>( result ) };
}

/// ...
template <typename State, typename Result>
class Executor
{
public:
    explicit Executor( Action<State, Result> action, State initialState = State() )
        : action_( std::move( action ) )
        , state_( std::move( initialState ) )
    {
        //
    }

    std::optional<Result> operator ()()
    {
        while ( !result_ && !process_() );
        return result_;
    }

private:
    bool process_()
    {
        return std::visit( [this] ( auto&& t )
        {
            return (*this)( std::forward<decltype( t )>( t ) );
        }, action_( state_ ) );
    }

    bool operator ()( YieldT&& )
    {
        return true;
    }

    bool operator ()( ContinueWithT<State>&& s )
    {
        state_ = std::move( s.state );
        return false;
    }

    bool operator ()( FinishWithT<Result>&& f )
    {
        result_ = std::move( f.result );
        return true;
    }

private:
    Action<State, Result> action_;
    State state_;
    std::optional<Result> result_;
};

} // namespace MR::StateMachine
