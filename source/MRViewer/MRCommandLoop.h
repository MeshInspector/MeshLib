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

    // Specify exciton in specific time of application start
    enum class StartPosition
    {
        AfterPluginInit, // executes during splash, after plugins init)
        AfterSplash, // executes after splash, to have valid main window context
        AfterWindowAppear // executes after window appeared to have valid opengl context
    };

    // This function setups main thread id, it should be called before any command
    MRVIEWER_API static void setMainThreadId( const std::thread::id& id );
    // Update state of command loop, only can rise
    MRVIEWER_API static void setState( StartPosition state );

    // Adds command to the end of command loop, can be performed from any thread
    // do not block, so be careful with lambda captures
    // note: state - specify exciton in specific time of application start
    MRVIEWER_API static void appendCommand( CommandFunc func, StartPosition state = StartPosition::AfterSplash );

    // If caller thread is main - instantly run command, otherwise add command to the end of loop with
    // StartPosition state = StartPosition::AfterSplash and blocks caller thread until command is done
    MRVIEWER_API static void runCommandFromGUIThread( CommandFunc func );

    // Execute all commands from loop
    MRVIEWER_API static void processCommands();
private:
    CommandLoop() = default;
    ~CommandLoop() = default;

    static CommandLoop& instance_();

    static void addCommand_( CommandFunc func, bool blockThread, StartPosition state );

    struct Command
    {
        CommandFunc func;
        StartPosition state{ StartPosition::AfterSplash };
        bool afterAppear{ false };
        std::condition_variable callerThreadCV;
        std::thread::id threadId;
    };

    StartPosition state_{ StartPosition::AfterPluginInit };

    std::thread::id mainThreadId_;
    std::queue<std::shared_ptr<Command>> commands_;
    std::mutex mutex_;
};

}