#pragma once
#include "exports.h"
#include "MRMesh/MRExpected.h"
#include <queue>
#include <functional>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace MR
{

// Additional command loop for external app control
class CommandLoop
{
public:
    using CommandFunc = std::function<void()>;

    // Specify execution in specific time of application start
    enum class StartPosition
    {
        AfterWindowInit,    // executes right after window is initialized
        AfterSplashAppear,  // executes after splash appeared
        AfterPluginInit,    // executes during splash, after plugins init)
        BeforeWindowAppear, // executes after splash is going to close, and just before main window is shown and have valid main window context
        AfterWindowAppear   // executes after window appeared to have valid opengl context
    };

    // This function setups main thread id, it should be called before any command
    MRVIEWER_API static void setMainThreadId( const std::thread::id& id );
    MRVIEWER_API static std::thread::id getMainThreadId();
    // Update state of command loop, only can rise
    MRVIEWER_API static void setState( StartPosition state );

    // Adds command to the end of command loop, can be performed from any thread
    // do not block, so be careful with lambda captures
    // note: state - specify execution in specific time of application start
    MRVIEWER_API static void appendCommand( CommandFunc func, StartPosition state = StartPosition::BeforeWindowAppear );

    // If caller thread is main - instantly run command, otherwise add command to the end of loop with
    // StartPosition state = StartPosition::AfterSplash and blocks caller thread until command is done
    // throws std::runtime_error if no loop can ever run the command: the queue is closed, it was never
    // started, or removeCommands dropped the command while the caller was waiting;
    // an exception thrown by func itself propagates to the caller as well;
    // call it only where an exception is already the error channel - python bindings and MCP tools -
    // and not from other C++ code, which must be exception-free: use appendCommand there
    MRVIEWER_API static void runCommandFromGUIThread( CommandFunc func );

    // Execute all commands from loop
    MRVIEWER_API static void processCommands();

    // Return true if loop is empty
    MRVIEWER_API static bool empty();

    // Clears the queue without executing the commands
    // if closeLoop is true, does not accept any new commands
    MRVIEWER_API static void removeCommands( bool closeLoop );

private:
    CommandLoop() = default;
    ~CommandLoop();

    static CommandLoop& instance_();

    // returns an error, rather than throwing or blocking forever, if no loop can ever run a blocking
    // command: the queue is closed, it was never started, or removeCommands dropped the command
    static Expected<void> addCommand_( CommandFunc func, bool blockThread, StartPosition state );

    struct Command
    {
        CommandFunc func;
        StartPosition state{ StartPosition::BeforeWindowAppear };
        std::condition_variable callerThreadCV;
        std::thread::id threadId;
        // set under CommandLoop::mutex_ just before func() is invoked; tells a blocked caller
        // that the main thread is executing its command rather than not having reached it yet
        bool started{ false };
        // set under CommandLoop::mutex_ once the command was executed or dropped;
        // the predicate a blocked caller waits on, see addCommand_
        bool done{ false };
        // set under CommandLoop::mutex_ in `removeCommands`: `done`, but never executed
        bool dropped{ false };
    };

    StartPosition state_{ StartPosition::AfterWindowInit };

    // if set then cannot accept new commands
    bool queueClosed_{ false }; // marked true in `removeCommands`
    std::thread::id mainThreadId_;
    std::queue<std::shared_ptr<Command>> commands_;
    std::mutex mutex_;
};

// Push a handful of empty commands onto the main thread so the Viewer advances a few frames,
// ensuring any UI state touched by a recent input (click, write, transform) is reflected before
// the caller's next observation.
// throws like runCommandFromGUIThread, so it has the same callers-only-from-python-and-MCP restriction
MRVIEWER_API void skipFramesAfterInput();

}