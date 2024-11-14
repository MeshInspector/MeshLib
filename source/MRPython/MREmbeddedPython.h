#pragma once

#include "exports.h"

#include <string>
#include <filesystem>

namespace MR
{

class MRPYTHON_CLASS EmbeddedPython
{
public:
    struct Config
    {
        bool siteImport{ true };
        std::string home;
        std::vector<std::string> argv;
    };
    static Config pythonConfig; // Set this once before running anything.

    static MRPYTHON_API bool isAvailable();

    // Returns true if the interpreter is busy running something.
    // If you try to run something else, your thread will block until it's done.
    [[nodiscard]] static bool nowRunning() { return instance_().state_.load() != State::idle; }

    // Returns false on failure.
    // If `onDoneAsync` is set, doesn't wait for the script to finish. Will call `onDoneAsync` asynchronously when done.
    static MRPYTHON_API bool runString( std::string pythonString, std::function<void( bool success )> onDoneAsync = nullptr );

    static MRPYTHON_API bool runScript( const std::filesystem::path& path );

    static MRPYTHON_API bool isPythonScript( const std::filesystem::path& path );
private:
    EmbeddedPython();
    ~EmbeddedPython() = default;

    bool init_();
    void ensureInterpreterThreadIsRunning_();

    static EmbeddedPython& instance_();
    bool available_{ false };

    enum class State { idle, starting, running, finishing };

    std::atomic<State> state_ = State::idle; // Could be non-atomic, but this makes things easier.
    std::string queuedSource_;
    bool lastRunSuccessful_ = false;
    std::function<void( bool success )> onDoneAsync_;
    std::mutex cvMutex_;
    std::condition_variable cv_; // It's could (?) be more efficient to have more CVs here, but one is simpler.

    std::thread interpreterThread_;
};

} //namespace MR
