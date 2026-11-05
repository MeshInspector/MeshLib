#include "MREmbeddedPython.h"

#include "MRMesh/MRFinally.h"
#include "MRMesh/MRString.h"
#include "MRMesh/MRStringConvert.h"
#include "MRMesh/MRSystemPath.h"
#include "MRPch/MRSpdlog.h"
#include "MRPython/MRPython.h"
#include "MRPython/MRUnifiedPythonStream.h"

#include <fstream>

#ifdef __linux__
// avoid macro redefinitions
#undef _POSIX_C_SOURCE
#undef _XOPEN_SOURCE
#endif
// The Debug builds use the Release Python: with `_DEBUG`, its headers would select the debug library and its object layout.
#if defined( _MSC_VER ) && defined( _DEBUG )
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

namespace MR
{

namespace
{

// The module `redirector`, the targets of `sys.stdout` and `sys.stderr`.
template <StreamType T>
PyObject* redirectorWrite( PyObject*, PyObject* arg )
{
    Py_ssize_t size = 0;
    const char* text = PyUnicode_AsUTF8AndSize( arg, &size );
    if ( !text )
        return nullptr;
    PythonStreamRedirector<T>().write( std::string( text, size_t( size ) ) );
    Py_RETURN_NONE;
}

template <StreamType T>
PyObject* redirectorFlush( PyObject*, PyObject* )
{
    PythonStreamRedirector<T>().flush();
    Py_RETURN_NONE;
}

PyMethodDef redirectorMethods[] = {
    { "write_stdout", redirectorWrite<Stdout>, METH_O, "write sys.stdout redirection." },
    { "flush_stdout", redirectorFlush<Stdout>, METH_NOARGS, "flush the stream" },
    { "write_stderr", redirectorWrite<Stderr>, METH_O, "write sys.stderr redirection." },
    { "flush_stderr", redirectorFlush<Stderr>, METH_NOARGS, "flush the stream" },
    { nullptr, nullptr, 0, nullptr },
};

PyModuleDef redirectorModule = { PyModuleDef_HEAD_INIT, "redirector", "Redirects Python's standard output and error output to the console.", -1, redirectorMethods, nullptr, nullptr, nullptr, nullptr };

PyObject* initRedirector()
{
    return PyModule_Create( &redirectorModule );
}

// The current Python error, with its traceback, cleared.
std::string fetchPythonError()
{
    PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
    PyErr_Fetch( &type, &value, &traceback );
    PyErr_NormalizeException( &type, &value, &traceback );
    MR_FINALLY{ Py_XDECREF( type ); Py_XDECREF( value ); Py_XDECREF( traceback ); };

    std::string ret;
    auto append = [&ret] ( PyObject* str )
    {
        if ( const char* s = str ? PyUnicode_AsUTF8( str ) : nullptr )
            ret += s;
        Py_XDECREF( str );
    };
    if ( PyObject* module = PyImport_ImportModule( "traceback" ) )
    {
        if ( PyObject* lines = PyObject_CallMethod( module, "format_exception", "OOO", type ? type : Py_None, value ? value : Py_None, traceback ? traceback : Py_None ) )
        {
            for ( Py_ssize_t i = 0, n = PyList_Check( lines ) ? PyList_Size( lines ) : 0; i < n; i++ )
            {
                PyObject* line = PyList_GetItem( lines, i );
                Py_XINCREF( line );
                append( line );
            }
            Py_DECREF( lines );
        }
        Py_DECREF( module );
    }
    if ( ret.empty() && value )
        append( PyObject_Str( value ) );
    PyErr_Clear();
    return ret.empty() ? "Unknown Python error." : ret;
}

// Runs `code` in the namespace of `__main__`, like `exec()`. Returns false with the error logged.
bool execPython( const std::string& code )
{
    PyObject* globals = PyModule_GetDict( PyImport_AddModule( "__main__" ) );
    if ( !PyDict_GetItemString( globals, "__builtins__" ) )
        PyDict_SetItemString( globals, "__builtins__", PyEval_GetBuiltins() );
    if ( !PyDict_GetItemString( globals, "__name__" ) )
    {
        PyObject* name = PyUnicode_FromString( "__main__" );
        PyDict_SetItemString( globals, "__name__", name );
        Py_XDECREF( name );
    }

    if ( PyObject* result = PyRun_String( code.c_str(), Py_file_input, globals, globals ) )
    {
        Py_DECREF( result );
        return true;
    }
    const auto error = fetchPythonError();
    spdlog::error( error );
    UnifiedPythonStream::get() << error; // add to unified python stream
    return false;
}

} // namespace

EmbeddedPython::Config EmbeddedPython::pythonConfig{};

bool EmbeddedPython::isAvailable()
{
    EmbeddedPython &self = instance_();
    return self.available_ && !self.shutdownCalled_;
}

void EmbeddedPython::shutdown()
{
    if ( !isAvailable() )
        return;

    EmbeddedPython &self = instance_();
    self.shutdownCalled_ = true;

    if ( !self.interpreterThread_.joinable() )
        return; // Nothing to do.

    { // Tell the thread to stop.
        spdlog::debug( "EmbeddedPython: shutdown, waiting for lock" );
        std::unique_lock guard( self.cvMutex_ );
        self.stopInterpreterThread_ = true;
        self.cv_.notify_all();
    }
    spdlog::debug( "EmbeddedPython: shutdown, join python thread" );
    self.interpreterThread_.join();
}

bool EmbeddedPython::runString( std::string pythonString, std::function<void( bool success )> onDoneAsync )
{
    if ( !isAvailable() )
        return false;

    EmbeddedPython &self = instance_();
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
    if ( !isAvailable() || !isPythonScript( path ) )
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
        c = (char)std::tolower( (unsigned char)c );

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
    shutdown();
}

bool EmbeddedPython::init_()
{
    if ( !isAvailable() )
        return true;

    PyConfig config;
    PyConfig_InitPythonConfig( &config );
    MR_FINALLY{ PyConfig_Clear( &config ); };

    config.site_import = pythonConfig.siteImport ? 1 : 0;

    auto failed = [] ( const PyStatus& status )
    {
        if ( !PyStatus_Exception( status ) )
            return false;
        const char* message = status.err_msg ? status.err_msg : "Unknown error while initializing Python.";
        spdlog::error( message );
        UnifiedPythonStream::get() << message; // add to unified python stream
        return true;
    };

    if ( !pythonConfig.home.empty() )
    {
        const auto homeW = utf8ToWide( pythonConfig.home.c_str() );
        if ( failed( PyConfig_SetString( &config, &config.home, homeW.c_str() ) ) )
            return false;
    }

    PyImport_AppendInittab( "redirector", initRedirector );

    PyStatus status;
    if ( pythonConfig.argv.empty() )
    {
        config.parse_argv = 0;
        config.install_signal_handlers = 0;
        status = PyConfig_SetBytesArgv( &config, 0, nullptr );
    }
    else
    {
        config.isolated = 1;
        std::vector<char *> argv;
        for ( auto& str : pythonConfig.argv )
            argv.push_back( str.data() );
        argv.push_back( nullptr ); // Unsure if needed, just in case.
        status = PyConfig_SetBytesArgv( &config, Py_ssize_t( argv.size() - 1 ), argv.data() );
    }
    if ( failed( status ) )
        return false;

    if ( failed( Py_InitializeFromConfig( &config ) ) )
        return false;

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
                { // Wait for python code.
                    cv_.wait( guard, [&]
                    {
                        return stopInterpreterThread_ || state_ == State::running;
                    } );
                    if ( stopInterpreterThread_ )
                        break;
                }

                lastRunSuccessful_ = false;

                static bool initOk = init_();
                if ( !initOk )
                {
                    spdlog::error( "Failed to initialize Python." );
                }
                else
                {
                    auto libDir = SystemPath::getPythonModulesDirectory();
                    auto libDirStr = utf8string( libDir );
                    MR::replaceInplace( libDirStr, "\\", "\\\\" ); // double protect first for C++ second for Python
                    std::string redirectScript =
                        "import sys\n"
                        "import redirector\n"
                        "class _Redirector:\n"
                        "    def __init__(self, write, flush):\n"
                        "        self.write = write\n"
                        "        self.flush = flush\n"
                        "sys.stdout = _Redirector(redirector.write_stdout, redirector.flush_stdout)\n"
                        "sys.stderr = _Redirector(redirector.write_stderr, redirector.flush_stderr)\n"
                        "del _Redirector\n"
                        "sys.path.insert(1,\"" + libDirStr + "\")\n";

                    lastRunSuccessful_ = execPython( redirectScript ) && execPython( queuedSource_ );
                    execPython( "sys.stdout.flush()\nsys.stderr.flush()" );

                    // We used to reset globals using `Py_FinalizeEx()`, but that beaks some modules (including ours at one point,
                    //   and apparently numpy too (https://stackoverflow.com/q/7676314/2752075), so it's not necessarily our bug).
                    PyDict_Clear( PyModule_GetDict( PyImport_AddModule( "__main__" ) ) );
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

            if ( Py_IsInitialized() )
                Py_FinalizeEx();
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
