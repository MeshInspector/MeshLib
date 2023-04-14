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
AsyncOrder::AsyncOrder()
{
    listenerThread_ = std::thread( [this] ()
    {
        MR::SetCurrentThreadName( "AsyncOrder timer thread" );
        while ( timer_.waitBlocking() != AsyncTimer::Event::Terminate )
        {
            auto cmd = loadCommand_();
            if ( cmd )
            {
                cmd();
                storeCommand_( {} );
            }
        }
    } );
}

AsyncOrder::~AsyncOrder()
{
    timer_.terminate();
    listenerThread_.join();
}

void AsyncOrder::order( const Time& time, Command command )
{
    timer_.setTime( time );
    storeCommand_( command );
}

void AsyncOrder::orderIfNotSet( const Time& time, Command command )
{
    if ( timer_.setTimeIfNotSet( time ) )
        storeCommand_( command );
}

void AsyncOrder::reset()
{
    timer_.resetTime();
    storeCommand_( {} );

}

MR::AsyncOrder::Command AsyncOrder::loadCommand_()
{
    std::unique_lock lock( cmdMutex_ );
    return command_;
}

void AsyncOrder::storeCommand_( Command command )
{
    std::unique_lock lock( cmdMutex_ );
    command_ = command;
}

#endif
} //namespace MR
