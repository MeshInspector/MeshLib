#pragma once

#include "exports.h"
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>

namespace MR
{

// the object to set timer from any thread and wait for alert time from another thread
class AsyncTimer
{
public: //to call from any requester thread
    // sets alert time, forgetting about previous time
    MRVIEWER_API void setTime( const std::chrono::time_point<std::chrono::system_clock> & time );
    // sets alert time only if it is not set yet
    MRVIEWER_API void setTimeIfNotSet( const std::chrono::time_point<std::chrono::system_clock> & time );
    // reset the timer
    MRVIEWER_API void resetTime();
    // orders the waiter thread to stop
    MRVIEWER_API void terminate();

public: //to call from waiter thread
    enum class Event
    {
        AlertTimeReached,
        Terminate
    };
    MRVIEWER_API Event waitBlocking();

private:
    std::mutex mutex_;
    std::condition_variable cvar_;
    std::optional<std::chrono::time_point<std::chrono::system_clock>> time_;
    bool terminating_ = false;
};

} //namespace MR
