#pragma once
#include "MRCommandLoop.h"

namespace MR
{

// Moves function func and copies/moves arguments to inner lambda object.
// After that pushes it to the event loop instead of immediate call.
template<typename F, typename... Args>
void pythonAppendOrRun( F func, Args&&... args )
{
    auto deferredAction = [funcLocal = std::move( func ), &...argsLocal = args]() mutable
    {
        funcLocal( std::forward<Args>( argsLocal )... );
    };
    MR::CommandLoop::runCommandFromGUIThread( std::move( deferredAction ) );
}

// Returns lambda which runs specified function `f` on commandLoop
// deferred instead of immediate call on current thread
// with signature of `f` and returns void
template<typename R, typename... Args>
auto pythonRunFromGUIThread( std::function<R( Args... )>&& f ) -> std::function<void( Args... )>
{
    return[fLocal = std::move( f )]( Args&&... args ) mutable
    {
        // fLocal must not be moved
        pythonAppendOrRun( fLocal, std::forward<Args>( args )... );
    };
}

template<typename F>
auto pythonRunFromGUIThread( F&& f )
{
    return pythonRunFromGUIThread( std::function( std::forward<F>( f ) ) );
}

template<typename R, typename T, typename... Args>
auto pythonRunFromGUIThread( R( T::* memFunction )( Args... ) )
{
    return pythonRunFromGUIThread( std::function<R( T*, Args... )>( std::mem_fn( memFunction ) ) );
}

}