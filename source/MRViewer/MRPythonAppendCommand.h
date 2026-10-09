#pragma once
#include "MRCommandLoop.h"
#include "MRPython/MRPybind11.h"

namespace MR
{

// CommandLoop::runCommandFromGUIThread for a binding: waits with the GIL released, since the GUI thread
// may need the GIL to get to the command, e.g. the macOS main thread still running Python code
template<typename F>
void pythonRunCommandFromGUIThread( F&& f )
{
    pybind11::gil_scoped_release gilRelease;
    CommandLoop::runCommandFromGUIThread( std::forward<F>( f ) );
}

// Moves function func and copies/moves arguments to inner lambda object.
// After that pushes it to the event loop instead of immediate call.
template<typename F, typename... Args>
void pythonAppendOrRun( F func, Args&&... args )
{
    auto deferredAction = [funcLocal = std::move( func ), &...argsLocal = args]() mutable
    {
        funcLocal( std::forward<Args>( argsLocal )... );
    };
    pythonRunCommandFromGUIThread( std::move( deferredAction ) );
}

// Returns lambda which runs specified function `f` on commandLoop
// deferred instead of immediate call on current thread
// with signature of `f` and returns void
template<typename R, typename... Args>
[[nodiscard]] auto pythonRunFromGUIThread( std::function<R( Args... )>&& f ) -> std::function<void( Args... )>
{
    return[fLocal = std::move( f )]( Args&&... args ) mutable
    {
        // fLocal must not be moved
        pythonAppendOrRun( fLocal, std::forward<Args>( args )... );
    };
}

template<typename F>
[[nodiscard]] auto pythonRunFromGUIThread( F&& f )
{
    return pythonRunFromGUIThread( std::function( std::forward<F>( f ) ) );
}

template<typename R, typename T, typename... Args>
auto pythonRunFromGUIThread( R( T::* memFunction )( Args... ) )
{
    return pythonRunFromGUIThread( std::function<R( T*, Args... )>( std::mem_fn( memFunction ) ) );
}

}
