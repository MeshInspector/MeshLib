#include "MRAsyncTimer.h"
#include "MRMesh/MRSystem.h"

namespace MR
{

void AsyncTimer::setTime( const std::chrono::time_point<std::chrono::system_clock> & time )
{
    std::unique_lock lock( mutex_ );
    time_ = time;
    cvar_.notify_one();
}

bool AsyncTimer::setTimeIfNotSet( const std::chrono::time_point<std::chrono::system_clock> & time )
{
    std::unique_lock lock( mutex_ );
    if ( !time_ )
    {
        time_ = time;
        cvar_.notify_one();
        return true;
    }
    return false;
}

void AsyncTimer::resetTime()
{
    std::unique_lock lock( mutex_ );
    time_.reset();
    //no need to notify
}

void AsyncTimer::terminate()
{
    std::unique_lock lock( mutex_ );
    terminating_ = true;
    cvar_.notify_one();
}

auto AsyncTimer::waitBlocking() -> Event
{
    std::unique_lock lock( mutex_ );
    for (;;)
    {
        if ( terminating_ )
            return Event::Terminate;
        if ( time_ )
        {
            auto waitTime = *time_;
            if ( std::cv_status::timeout == cvar_.wait_until( lock, waitTime ) )
            {
                if ( time_ && *time_ == waitTime ) //nothing has changed
                {
                    time_.reset();
                    return Event::AlertTimeReached;
                }
            }
            continue;
        }
        cvar_.wait( lock );
    }
}

#ifndef __EMSCRIPTEN__
AsyncRequest::AsyncRequest()
{
    listenerThread_ = std::thread( [this] ()
    {
        MR::SetCurrentThreadName( "AsyncRequest timer thread" );
        while ( timer_.waitBlocking() != AsyncTimer::Event::Terminate )
        {
            std::unique_lock lock( cmdMutex_ );
            if ( command_ )
            {
                command_();
                command_ = {};
            }
        }
    } );
}

AsyncRequest::~AsyncRequest()
{
    timer_.terminate();
    listenerThread_.join();
}

void AsyncRequest::request( const Time& time, Command command )
{
    timer_.setTime( time );
    storeCommand_( command );
}

void AsyncRequest::requestIfNotSet( const Time& time, Command command )
{
    if ( timer_.setTimeIfNotSet( time ) )
        storeCommand_( command );
}

void AsyncRequest::reset()
{
    timer_.resetTime();
    storeCommand_( {} );

}

MR::AsyncRequest::Command AsyncRequest::loadCommand_()
{
    std::unique_lock lock( cmdMutex_ );
    return command_;
}

void AsyncRequest::storeCommand_( Command command )
{
    std::unique_lock lock( cmdMutex_ );
    command_ = command;
}

#endif
} //namespace MR
