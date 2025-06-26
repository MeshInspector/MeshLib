#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include <GLFW/glfw3.h>
#include <assert.h>

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
        return addCommand_( func, true, StartPosition::AfterSplashHide );
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
    for ( auto& cmdToNotify : commandsToNotifyAtTheEnd )
        cmdToNotify->callerThreadCV.notify_one();
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
        cmd->callerThreadCV.wait( lock );

        if ( exception )
            std::rethrow_exception( exception );
    }
}

}
