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
        cmd->started = true;
        lock.unlock();

        cmd->func();
        assert( inst.mainThreadId_ == std::this_thread::get_id() );
        if ( cmd->threadId != inst.mainThreadId_ )
            commandsToNotifyAtTheEnd.emplace_back( std::move( cmd ) );
    }
    if ( !commandsToNotifyAtTheEnd.empty() )
    {
        {
            std::unique_lock<std::mutex> lock( inst.mutex_ );
            for ( auto& cmdToNotify : commandsToNotifyAtTheEnd )
                cmdToNotify->done = true;
        }
        // notify with the mutex released, otherwise every woken caller at once blocks on re-acquiring it;
        // the shared_ptr-s here keep the commands, and so the condition variables, alive
        for ( auto& cmdToNotify : commandsToNotifyAtTheEnd )
            cmdToNotify->callerThreadCV.notify_one();
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
    std::vector<std::shared_ptr<Command>> droppedCommands;
    {
        std::unique_lock<std::mutex> lock( inst.mutex_ );
        inst.queueClosed_ = closeLoop;
        while ( !inst.commands_.empty() )
        {
            auto cmd = std::move( inst.commands_.front() );
            inst.commands_.pop();
            cmd->done = true;
            droppedCommands.emplace_back( std::move( cmd ) );
        }
    }
    // notify and log with the mutex released, see processCommands
    for ( auto& cmd : droppedCommands )
        cmd->callerThreadCV.notify_one();
    spdlog::debug( "CommandLoop::removeCommands(): dropped {} command(s)", droppedCommands.size() );
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
        // wait on `done`, not on a bare notification: a spurious wakeup would otherwise return while
        // `exception` above - a local of this frame - is still captured by the not yet executed command
        // wait with a timeout: the event posted above can be lost or never woken on, and an untimed wait
        // turns that into an indefinite block with nothing in the log to attribute it to
        using namespace std::chrono;
        constexpr auto cMaxWarnStep = seconds( 300 );
        const auto startTime = steady_clock::now();
        auto warnStep = seconds( 5 );  // doubles up to cMaxWarnStep, not to flood the log
        auto nextWarn = startTime + warnStep;
#ifdef __linux__
        // Re-post the wakeup only on Linux: there the event posted above can be genuinely lost, and
        // re-posting it is what gets the main thread out of glfwWaitEvents(). Elsewhere the event is
        // delivered reliably, so the main thread is busy or blocked for another reason and one more
        // event would change nothing.
        // Unlike the warning it is cheap and silent, so it runs at a flat and much shorter period,
        // making a lost event cost about a second instead of the first warning step.
        constexpr auto cRepostPeriod = seconds( 1 );
        auto nextRepost = startTime + cRepostPeriod;
#endif
        for ( ;; )
        {
            auto nextWake = nextWarn;
#ifdef __linux__
            if ( !cmd->started ) // no re-posts can help once the command is executing
                nextWake = std::min( nextWake, nextRepost );
#endif
            if ( cmd->callerThreadCV.wait_until( lock, nextWake, [&cmd] { return cmd->done; } ) )
                break;

            const auto now = steady_clock::now();
            const bool started = cmd->started; // read under the lock
            const bool warn = now >= nextWarn;
            if ( warn )
            {
                warnStep = std::min( warnStep * 2, cMaxWarnStep );
                nextWarn = now + warnStep;
            }
#ifdef __linux__
            // a command already inside func() is not waiting for an event, so re-posting cannot help it
            const bool repost = now >= nextRepost && !started;
            if ( now >= nextRepost )
                nextRepost = now + cRepostPeriod;
#endif
            // release the mutex for the body: neither sink I/O nor posting an event must hold up processCommands
            lock.unlock();
            if ( warn )
            {
                // separate calls, spdlog needs the format string to be a compile-time constant
                const auto waited = duration_cast<seconds>( now - startTime ).count();
                if ( started )
                    spdlog::warn( "CommandLoop::addCommand_: the main thread is still executing a queued command, {} seconds so far", waited );
                else
                    spdlog::warn( "CommandLoop::addCommand_: the main thread has not started a queued command in {} seconds", waited );
            }
#ifdef __linux__
            if ( repost )
                getViewerInstance().postEmptyEvent();
#endif
            lock.lock();
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
