#include "MREmbeddedPython.h"
#include "MRPython.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRPch/MRSpdlog.h"
#include "MRUnifiedPythonStream.h"

#include <pybind11/embed.h>

#include <fstream>

namespace MR
{

EmbeddedPython::Config EmbeddedPython::pythonConfig{};

bool EmbeddedPython::isAvailable()
{
    return instance_().available_;
}

bool EmbeddedPython::runString( std::string pythonString, std::function<void( bool success )> onDoneAsync )
{
    EmbeddedPython &self = instance_();
    if ( !self.available_ )
        return false;

    self.ensureInterpreterThreadIsRunning_();

    // Negotiate with the interpreter thread.
    std::unique_lock guard( self.cvMutex_ );
    self.cv_.wait( guard, [&]{ return self.state_ == State::idle; } );
    self.queuedSource_ = std::move( pythonString );
    self.state_ = State::running;
    self.onDoneAsync_ = std::move( onDoneAsync );
    self.cv_.notify_all();
    if ( self.onDoneAsync_ )
    {
        return true;
    }
    else
    {
        self.cv_.wait( guard, [&]{ return self.state_ == State::finishing; } );
        self.state_ = State::idle;
        self.cv_.notify_all();
        return self.lastRunSuccessful_;
    }
}

bool EmbeddedPython::runScript( const std::filesystem::path& path )
{
    if ( !instance_().available_ || !isPythonScript( path ) )
        return false;

    std::ifstream ifs( path );
    std::ostringstream oss;
    oss << ifs.rdbuf();
    ifs.close();
    std::string str = oss.str();
    return runString( str );
}

bool EmbeddedPython::isPythonScript( const std::filesystem::path& path )
{
    std::error_code ec;
    if ( !std::filesystem::exists( path, ec ) )
        return false;
    if ( !std::filesystem::is_regular_file( path, ec ) )
        return false;

    auto ext = utf8string( path.extension() );
    for ( auto& c : ext )
        c = (char) tolower( c );

    if ( ext != ".py" )
        return false;

    return true;
}

EmbeddedPython::EmbeddedPython()
{
    available_ = !Py_IsInitialized();
}

EmbeddedPython::~EmbeddedPython()
{
    if ( !interpreterThread_.joinable() )
        return; // Nothing to do.

    { // Tell the thread to stop.
        stopInterpreterThread_ = true;
        cv_.notify_all();
    }

    interpreterThread_.join();
}

bool EmbeddedPython::init_()
{
    EmbeddedPython &self = instance_();
    if ( !self.available_ )
        return true;

    PyConfig config;
    PyConfig_InitPythonConfig( &config );
    MR_FINALLY { PyConfig_Clear( &config ); };

    config.site_import = pythonConfig.siteImport ? 1 : 0;

    if ( !pythonConfig.home.empty() )
    {
        const auto homeW = utf8ToWide( pythonConfig.home.c_str() );
        PyConfig_SetString( &config, &config.home, homeW.c_str() );
    }

    for ( const auto& mod : PythonExport::instance().modules() )
        PyImport_AppendInittab( mod.first.c_str(), mod.second.initFncPointer );

    PyStatus status{};
    if ( pythonConfig.argv.empty() )
    {
        config.parse_argv = 0;
        config.install_signal_handlers = 0;
        status = PyConfig_SetBytesArgv( &config, 0, NULL );
    }
    else
    {
        config.isolated = 1;
        std::vector<char *> argv;
        for ( auto& str : pythonConfig.argv )
            argv.push_back( str.data() );
        argv.push_back( nullptr ); // Unsure if needed, just in case.
        status = PyConfig_SetBytesArgv( &config, argv.size() - 1, argv.data() );
    }

    if ( PyStatus_Exception( status ) )
    {
        spdlog::error( status.err_msg );
        UnifiedPythonStream::get() << status.err_msg; // add to unified python stream
        return false;
    }

    status = Py_InitializeFromConfig( &config );
    if ( PyStatus_Exception( status ) )
    {
        spdlog::error( status.err_msg );
        UnifiedPythonStream::get() << status.err_msg; // add to unified python stream
        return false;
    }

    return true;
}

void EmbeddedPython::ensureInterpreterThreadIsRunning_()
{
    [[maybe_unused]] static const auto once = [&]{
        // Start the interpreter thread.
        interpreterThread_ = std::thread( [this]
        {
            std::unique_lock guard( cvMutex_ );

            while ( true )
            {
                { // Wait for source code.
                    bool stop = false;
                    cv_.wait( guard, [&]
                    {
                        if ( stopInterpreterThread_ )
                        {
                            stop = true;
                            return true;
                        }

                        return state_ == State::running;
                    } );
                    if ( stop )
                        break;
                }

                lastRunSuccessful_ = false;

                static bool initOk = init_();
                if ( !initOk )
                {
                    spdlog::error( "Failied to initialize Python." );
                }
                else
                {
                    try
                    {
                        auto libDir = SystemPath::getPythonModulesDirectory();
                        auto libDirStr = utf8string( libDir );
                        MR::replaceInplace( libDirStr, "\\", "\\\\" ); // double protect first for C++ second for Python
                        // Create an empty dictionary that will function as a namespace.
                        std::string redirectScript =
                            "import sys\n"
                            "import redirector\n"
                            "sys.stdout = redirector.stdout()\n"
                            "sys.stderr = redirector.stderr()\n"
                            "sys.path.insert(1,\"" + libDirStr + "\")\n";
                        pybind11::exec( redirectScript.c_str() );

                        // Execute code
                        pybind11::exec( queuedSource_.c_str() );

                        pybind11::exec( "sys.stdout.flush()\nsys.stderr.flush()" );

                        lastRunSuccessful_ = true;
                    }
                    catch ( std::exception& e )
                    {
                        spdlog::error( e.what() );
                        UnifiedPythonStream::get() << e.what(); // add to unified python stream
                    }
                    catch ( ... )
                    {
                        spdlog::error( "Unknown exception while executing a Python script." );
                    }

                    try
                    {
                        // We used to reset globals using `pybind11::finalize_interpreter()`, but that beaks some modules (including ours at one point,
                        //   and apparently numpy too (https://stackoverflow.com/q/7676314/2752075), so it's not necessarily our bug).
                        pybind11::globals().clear();
                    }
                    catch ( ... )
                    {
                        spdlog::error( "Unable to reset the global variables after the script because of an exception." );
                    }
                }

                { // Signal that we're done.
                    if ( onDoneAsync_ )
                    {
                        onDoneAsync_( lastRunSuccessful_ );
                        onDoneAsync_ = nullptr;
                        state_ = State::idle;
                    }
                    else
                    {
                        state_ = State::finishing;
                    }
                    cv_.notify_all();
                }
            }

            pybind11::finalize_interpreter();
        } );

        return nullptr;
    }();
}

EmbeddedPython& EmbeddedPython::instance_()
{
    static EmbeddedPython instance;
    return instance;
}

} //namespace MR
