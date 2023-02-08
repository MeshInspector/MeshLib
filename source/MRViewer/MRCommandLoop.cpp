#include "MRCommandLoop.h"
#include "MRViewer.h"
#include "MRPch/MRSpdlog.h"
#include <GLFW/glfw3.h>
#include <assert.h>

namespace MR
{

void CommandLoop::setMainThreadId( const std::thread::id& id )
{
    auto& inst = instance_();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    inst.mainThreadId_ = id;
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
        return addCommand_( func, true, StartPosition::AfterSplash );
    else
        return func();
}

void CommandLoop::processCommands()
{
    auto& inst = instance_();
    std::shared_ptr<Command> refCommand;
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
            cmd->callerThreadCV.notify_one();
    }
}

CommandLoop& CommandLoop::instance_()
{
    static CommandLoop commadLoop_;
    return commadLoop_;
}

void CommandLoop::addCommand_( CommandFunc func, bool blockThread, StartPosition state )
{
    auto& inst = instance_();
    std::shared_ptr<Command> cmd = std::make_shared<Command>();
    cmd->state = state;
    cmd->func = func;
    cmd->threadId = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    inst.commands_.push( cmd );

    getViewerInstance().postEmptyEvent();
    if ( blockThread )
        cmd->callerThreadCV.wait( lock );
}

}