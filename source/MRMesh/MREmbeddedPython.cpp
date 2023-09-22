#include "MREmbeddedPython.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PYTHON )
#include "MRPython.h"
#include "MRStringConvert.h"
#include "MRPch/MRSpdlog.h"
#include <pybind11/embed.h>
#include <fstream>

namespace MR
{

void EmbeddedPython::init()
{
    if ( !instance_().available_ || isInitialized() )
        return;

    for ( const auto& mod : PythonExport::instance().modules() )
        PyImport_AppendInittab( mod.first.c_str(), mod.second.initFncPointer );

    pybind11::initialize_interpreter( false );
}

bool EmbeddedPython::isAvailable()
{
    return instance_().available_;
}

bool EmbeddedPython::isInitialized()
{
    if ( !instance_().available_ )
        return false;
    return (bool)Py_IsInitialized();
}

void EmbeddedPython::finalize()
{
    if ( !instance_().available_ )
        return;
    pybind11::finalize_interpreter();
}

bool EmbeddedPython::setupArgv( int argc, char** argv )
{
    if ( !instance_().available_ )
        return false;
    PyStatus status;

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.isolated = 1;

    // Implicitly preinitialize Python (in isolated mode)
    status = PyConfig_SetBytesArgv(&config, argc, argv);
    if (PyStatus_Exception(status))
        return false;

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status))
        return false;
    return true;
}

bool EmbeddedPython::runString( const std::string& pythonString )
{
    if ( !instance_().available_ )
        return false;
    namespace python = pybind11;
    bool success = true;
    try
    {
        // Create an empty dictionary that will function as a namespace.
        std::string redirectScript =
            "import sys\n"
            "import redirector\n"
            "sys.stdout = redirector.stdout()\n"
            "sys.stderr = redirector.stderr()";
        python::exec( redirectScript.c_str() );

        // Execute code
        python::exec( pythonString.c_str() );
    }
    catch ( std::exception& e )
    {
        success = false;
        spdlog::error( e.what() );
    }
    return success;
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

EmbeddedPython& EmbeddedPython::instance_()
{
    static EmbeddedPython instance;
    return instance;
}

} //namespace MR
#endif
