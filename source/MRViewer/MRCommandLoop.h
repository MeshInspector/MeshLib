#pragma once
#include "exports.h"
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

    // This function setups main thread id, it should be called before any command
    MRVIEWER_API static void setMainThreadId( const std::thread::id& id );
    // Notify event loop that window has appeared
    MRVIEWER_API static void setWindowAppeared();

    // Adds command to the end of command loop, can be performed from any thread
    // do not block, so be careful with lambda captures
    MRVIEWER_API static void appendCommand( CommandFunc func );
    MRVIEWER_API static void appendCommandAfterWindowAppear( CommandFunc func );

    // If caller thread is main - instantly run command, otherwise add command to the end of loop 
    // and blocks caller thread until command is done
    MRVIEWER_API static void runCommandFromGUIThread( CommandFunc func );

    // Execute all commands from loop
    MRVIEWER_API static void processCommands();
private:
    CommandLoop() = default;
    ~CommandLoop() = default;

    static CommandLoop& instance_();

    static void addCommand_( CommandFunc func, bool blockThread, bool afterAppear );

    struct Command
    {
        CommandFunc func;
        bool afterAppear{ false };
        std::condition_variable callerThreadCV;
        std::thread::id threadId;
    };

    bool windowAppeared_{ false };

    std::thread::id mainThreadId_;
    std::queue<std::shared_ptr<Command>> commands_;
    std::mutex mutex_;
};

}