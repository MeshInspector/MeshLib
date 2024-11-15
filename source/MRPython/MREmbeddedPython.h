#pragma once

#include "exports.h"

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
    MRPYTHON_API static Config pythonConfig; // Set this once before running anything.

    static MRPYTHON_API bool isAvailable();

    // Returns true if the interpreter is busy running something.
    // If you try to run something else, your thread will block until it's done.
    [[nodiscard]] static bool nowRunning() { return instance_().state_.load() != State::idle; }

    // Returns false on failure.
    // If `onDoneAsync` is set, doesn't wait for the script to finish.
    // Will call `onDoneAsync` asynchronously when done (from the Python interpreter thread).
    static MRPYTHON_API bool runString( std::string pythonString, std::function<void( bool success )> onDoneAsync = nullptr );

    static MRPYTHON_API bool runScript( const std::filesystem::path& path );

    static MRPYTHON_API bool isPythonScript( const std::filesystem::path& path );
private:
    EmbeddedPython();
    EmbeddedPython( const EmbeddedPython& ) = delete;
    EmbeddedPython& operator=( const EmbeddedPython& ) = delete;
    ~EmbeddedPython();

    bool init_();
    void ensureInterpreterThreadIsRunning_();

    MRPYTHON_API static EmbeddedPython& instance_();
    bool available_{ false };

    enum class State
    {
        idle, // Waiting for a script to run.
        running, // Interpreter is running.
        finishing, // Interpreter is done, waiting for the submitter thread to read the result.
    };

    std::atomic<State> state_ = State::idle; // Making this atomic allows `nowRunning()` to read this without locking the mutex.
    std::string queuedSource_;
    bool lastRunSuccessful_ = false;
    std::function<void( bool success )> onDoneAsync_;
    std::mutex cvMutex_;
    std::condition_variable cv_; // It's could (?) be more efficient to have more CVs here, but one is simpler.

    // We maintain ONE persistent thread that runs all python scripts, and persists while the program runs.
    // This seems to be the safest option, I had issues otherwise. We need everything Python-related to happen in the same thread,
    // and we also need to not finalize-and-recreate the interpreter while the program runs because that breaks our generated bindings
    // (which may or may not be possible to fix in the bindings, but it's easier not to, and the manual even advises that
    // some modules can break if you recreate the interpeter: https://docs.python.org/3/c-api/init.html#c.Py_FinalizeEx).

    std::thread interpreterThread_;

    std::atomic_bool stopInterpreterThread_ = false;
};

} //namespace MR
