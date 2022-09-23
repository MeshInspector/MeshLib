#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PYTHON )
#include "MRPython.h"
#include <vector>
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

}
#endif
