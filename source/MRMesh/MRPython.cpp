#include "MRPython.h"
#include "MRStringConvert.h"
#include <pybind11/embed.h>
#include <spdlog/spdlog.h>
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>

template<StreamType T>
struct NumWritten
{
    static inline int counter = 0;
};

template<StreamType T>
void PythonStreamRedirector<T>::write( const std::string& text )
{
    ++NumWritten<T>::counter;
    if constexpr ( T == Stdout )
        std::cout << text;
    else
        std::cerr << text;
}

template<StreamType T>
int PythonStreamRedirector<T>::getNumWritten() 
{
    return NumWritten<T>::counter;
}

template class PythonStreamRedirector<Stdout>;
template class PythonStreamRedirector<Stderr>;

PYBIND11_MODULE( redirector, m )
{
    pybind11::class_<StdoutPyRedirector>( m, "stdout",
                              "This class redirects python's standard output to the console.    ")
        .def( pybind11::init<>(), "initialize the redirector." )
        .def( pybind11::init( [] () { return std::make_unique<StdoutPyRedirector>(); } ), "initialize the redirector." )
        .def( "write", &StdoutPyRedirector::write, "write sys.stdout redirection." )
        .def( "flush", &StdoutPyRedirector::flush, "empty func" );

    pybind11::class_<StderrPyRedirector>( m, "stderr",
                              "This class redirects python's error output to the console." )
        .def( pybind11::init<>(), "initialize the redirector." )
        .def( pybind11::init( [] () { return std::make_unique<StderrPyRedirector>(); } ), "initialize the redirector." )
        .def( "write", &StderrPyRedirector::write, "write sys.stderr redirection." )
        .def( "flush", &StderrPyRedirector::flush, "empty func" );
}
static MR::PythonFunctionAdder redirector_init_( "redirector", &PyInit_redirector );

namespace MR
{

PythonExport& PythonExport::instance()
{
    static PythonExport instance_;
    return instance_;
}

PythonFunctionAdder::PythonFunctionAdder( const std::string& moduleName, std::function<void( pybind11::module_& m )> func )
{
    PythonExport::instance().addFunc( moduleName, func );
}

PythonFunctionAdder::PythonFunctionAdder( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) )
{
    PythonExport::instance().setInitFuncPtr( moduleName, initFncPointer );
}

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
    std::vector<std::wstring> wargv( argc );
    std::vector<wchar_t*> wargvPtr( argc );
    for ( int i = 0; i < argc; ++i )
    {
        wargv[i] = utf8ToWide( argv[i] );
        wargvPtr[i] = wargv[i].data();
    }
    PySys_SetArgv( argc, wargvPtr.data() );
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
    catch ( std::runtime_error& e )
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

    auto ext = path.extension().u8string();
    for ( auto& c : ext )
        c = (char) tolower( c );

    if ( ext != u8".py" )
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

}

