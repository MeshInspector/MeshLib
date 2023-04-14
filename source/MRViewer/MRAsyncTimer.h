#pragma once

#include "exports.h"
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <optional>
#ifndef __EMSCRIPTEN__
#include <thread>
#include <functional>
#include <memory>
#endif

namespace MR
{

using Time = std::chrono::time_point<std::chrono::system_clock>;

// the object to set timer from any thread and wait for alert time from another thread
class MRVIEWER_CLASS AsyncTimer
{
public: //to call from any requester thread
    // sets alert time, forgetting about previous time
    MRVIEWER_API void setTime( const Time& time );
    // sets alert time only if it is not set yet
    // returns true if timer set
    MRVIEWER_API bool setTimeIfNotSet( const Time& time );
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
    std::optional<Time> time_;
    bool terminating_ = false;
};

#ifndef __EMSCRIPTEN__
// Complete given command in given time
// may know only one command at a time
// terminate listener in destructor
class MRVIEWER_CLASS AsyncRequest
{
public:
    MRVIEWER_API AsyncRequest();
    MRVIEWER_API ~AsyncRequest();
    using Command = std::function<void()>;

    // requests command execution, forgetting about previous command
    // note that command will be executed from the listener thread
    MRVIEWER_API void request( const Time& time, Command command );

    // requests command execution only if no request waiting
    // note that command will be executed from the listener thread
    MRVIEWER_API void requestIfNotSet( const Time& time, Command command );

    // clears command
    MRVIEWER_API void reset();
private:
    std::thread listenerThread_;
    AsyncTimer timer_;

    Command loadCommand_();
    void storeCommand_( Command command );

    std::mutex cmdMutex_;
    Command command_;

};
#endif
} //namespace MR
