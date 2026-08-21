#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include <GLFW/glfw3.h>
#include <assert.h>
#include <algorithm>
#include <chrono>

namespace MR
{

CommandLoop::~CommandLoop()
{
    spdlog::debug( "CommandLoop::~CommandLoop(): queue size={}", commands_.size() );
    assert( commands_.empty() );
}

void CommandLoop::setMainThreadId( const std::thread::id& id )
{
    auto& inst = instance_();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    inst.mainThreadId_ = id;
}

std::thread::id CommandLoop::getMainThreadId()
{
    return instance_().mainThreadId_;
}

void CommandLoop::setState( StartPosition state )
{
    auto& inst = instance_();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    if ( state < inst.state_ )
    {
        spdlog::warn( "Downgrade CommandLoop state is not possible" );
        return;
    }
    inst.state_ = state;
}

void CommandLoop::appendCommand( CommandFunc func, StartPosition pos )
{
    addCommand_( func, false, pos );
}

void CommandLoop::runCommandFromGUIThread( CommandFunc func )
{
    bool blockThread = instance_().mainThreadId_ != std::this_thread::get_id();
    if ( blockThread )
        return addCommand_( func, true, StartPosition::BeforeWindowAppear );
    else
        return func();
}

void CommandLoop::processCommands()
{
    auto& inst = instance_();
    using CmdPtr = std::shared_ptr<Command>;
    CmdPtr refCommand;
    std::vector<CmdPtr> commandsToNotifyAtTheEnd; // notify out of loop to be sure that next blocking cmd will be executed in the next frame
    for ( ; ;)
    {
        std::unique_lock<std::mutex> lock( inst.mutex_ );
        if ( inst.commands_.empty() )
            break;
        auto cmd = inst.commands_.front();
        if ( inst.state_ < cmd->state )
        {
            if ( refCommand == cmd )
                break;
            if ( !refCommand )
                refCommand = cmd;
            inst.commands_.push( cmd );
            inst.commands_.pop();
            continue;
        }
        inst.commands_.pop();
        lock.unlock();

        cmd->func();
        assert( inst.mainThreadId_ == std::this_thread::get_id() );
        if ( cmd->threadId != inst.mainThreadId_ )
            commandsToNotifyAtTheEnd.emplace_back( std::move( cmd ) );
    }
    if ( !commandsToNotifyAtTheEnd.empty() )
    {
        std::unique_lock<std::mutex> lock( inst.mutex_ );
        for ( auto& cmdToNotify : commandsToNotifyAtTheEnd )
        {
            cmdToNotify->done = true;
            cmdToNotify->callerThreadCV.notify_one();
        }
    }
}

bool CommandLoop::empty()
{
    auto& inst = instance_();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    return inst.commands_.empty();
}

void CommandLoop::removeCommands( bool closeLoop )
{
    auto& inst = instance_();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    inst.queueClosed_ = closeLoop;
    while ( !inst.commands_.empty() )
    {
        auto cmd = std::move( inst.commands_.front() );
        inst.commands_.pop();
        cmd->done = true;
        cmd->callerThreadCV.notify_one();
    }
    spdlog::debug( "CommandLoop::removeCommands(): queue size={}", inst.commands_.size() );
}

CommandLoop& CommandLoop::instance_()
{
    static CommandLoop commadLoop_;
    return commadLoop_;
}

void CommandLoop::addCommand_( CommandFunc func, bool blockThread, StartPosition state )
{
    std::exception_ptr exception;
    if ( blockThread )
    {
        // Adjust the `func` to store thrown exceptions.
        func = [next = std::move( func ), &exception]
        {
            try
            {
                next();
            }
            catch ( ... )
            {
                exception = std::current_exception();
            }
        };
    }

    auto& inst = instance_();
    std::shared_ptr<Command> cmd = std::make_shared<Command>();
    cmd->state = state;
    cmd->func = func;
    cmd->threadId = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    if ( inst.queueClosed_ )
    {
        spdlog::debug( "CommandLoop::addCommand_: cannot accept new command because it is closed" );
        return;
    }
    inst.commands_.push( cmd );

    getViewerInstance().postEmptyEvent();
    if ( blockThread )
    {
        // Wait on `done`, not on a bare notification: a spurious wakeup would otherwise return
        // from a command that has not run yet, with `exception` above - a local of this frame -
        // still captured by it.
        // And wait with a timeout, because the main thread can be parked in glfwWaitEvents()
        // having never woken on the event posted above; an untimed wait makes that an indefinite
        // and silent block, while the warning below leaves a trace of it in the log.
        // The step doubles up to cMaxWaitStep, so a main thread that is merely busy for a long
        // time - or a process that never runs the loop at all - does not flood the log.
        constexpr auto cMaxWaitStep = std::chrono::seconds( 300 );
        auto waitStep = std::chrono::seconds( 5 );
        auto waited = std::chrono::seconds( 0 );
        while ( !cmd->callerThreadCV.wait_for( lock, waitStep, [&cmd] { return cmd->done; } ) )
        {
            waited += waitStep;
            waitStep = std::min( waitStep * 2, cMaxWaitStep );
            spdlog::warn( "CommandLoop::addCommand_: the main thread has not run a queued command in {} seconds",
                waited.count() );
#ifdef __linux__
            // Re-post the wakeup only on Linux: there the event posted above can be genuinely lost,
            // and re-posting it is what gets the main thread out of glfwWaitEvents(). Elsewhere the
            // event is delivered reliably, so the main thread is busy or blocked for another reason
            // and one more event would change nothing.
            getViewerInstance().postEmptyEvent();
#endif
        }

        if ( exception )
            std::rethrow_exception( exception );
    }
}

void skipFramesAfterInput()
{
    for ( int i = 0; i < getViewerInstance().forceRedrawMinimumIncrementAfterEvents; ++i )
        CommandLoop::runCommandFromGUIThread( [] {} );
}

}
