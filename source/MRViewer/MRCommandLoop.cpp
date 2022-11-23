#include "MRCommandLoop.h"
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

void CommandLoop::appendCommand( CommandFunc func )
{
    addCommand_( func, false );
}

void CommandLoop::runCommandFromGUIThread( CommandFunc func )
{
    bool blockThread = instance_().mainThreadId_ != std::this_thread::get_id();
    if ( blockThread )
        return addCommand_( func, true );
    else
        return func();
}

void CommandLoop::processCommands()
{
    auto& inst = instance_();
    for ( ; ;)
    {
        std::unique_lock<std::mutex> lock( inst.mutex_ );
        if ( inst.commands_.empty() )
            break;
        auto cmd = std::move( inst.commands_.front() );
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

void CommandLoop::addCommand_( CommandFunc func, bool blockThread )
{
    auto& inst = instance_();
    std::shared_ptr<Command> cmd = std::make_shared<Command>();
    cmd->func = func;
    cmd->threadId = std::this_thread::get_id();
    std::unique_lock<std::mutex> lock( inst.mutex_ );
    inst.commands_.push( cmd );

    glfwPostEmptyEvent();
    if ( blockThread )
        cmd->callerThreadCV.wait( lock );
}

std::thread::id CommandLoop::getMainThreadId() 
{
    return instance_().mainThreadId_;
}

}