#include "MRAsyncTimer.h"

namespace MR
{

void AsyncTimer::setTime( const std::chrono::time_point<std::chrono::system_clock> & time )
{
    std::unique_lock lock( mutex_ );
    time_ = time;
    cvar_.notify_one();
}

void AsyncTimer::setTimeIfNotSet( const std::chrono::time_point<std::chrono::system_clock> & time )
{
    std::unique_lock lock( mutex_ );
    if ( !time_ )
    {
        time_ = time;
        cvar_.notify_one();
    }
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

} //namespace MR
