#pragma once

#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <tl/expected.hpp>
#include <functional>
#include <filesystem>
#include <unordered_map>

#define MR_INIT_PYTHON_MODULE( moduleName ) MR_INIT_PYTHON_MODULE_PRECALL( moduleName, [](){} )

#define MR_INIT_PYTHON_MODULE_PRECALL( moduleName, precall )\
PYBIND11_MODULE( moduleName, m )\
{\
    precall();\
    auto& adders = MR::PythonExport::instance().functions( #moduleName );\
    for ( auto& f : adders )\
        f( m );\
}\
static MR::PythonFunctionAdder moduleName##_init_( #moduleName, &PyInit_##moduleName );

#define MR_ADD_PYTHON_FUNCTION( moduleName , name , func , description ) \
    static MR::PythonFunctionAdder name##_adder_( #moduleName, [](pybind11::module_& m){ m.def(#name, func, description);} );

#define MR_ADD_PYTHON_CUSTOM_DEF( moduleName , name , ... ) \
_Pragma("warning(push)") \
_Pragma("warning(disable:4459)") \
    static MR::PythonFunctionAdder name##_adder_( #moduleName, __VA_ARGS__ ); \
_Pragma("warning(pop)")

// !!! Its important to add vec after adding type
// otherwise embedded python will not be able to re-import module (due to some issues with vector types in pybind11)
#define MR_ADD_PYTHON_VEC( moduleName , name , type)\
PYBIND11_MAKE_OPAQUE( std::vector<type> )\
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name, [] (pybind11::module_& m)\
{\
    using vecType = std::vector<type>;\
    pybind11::bind_vector<vecType>(m, #name ).\
        def( "empty", &vecType::empty ).\
        def( "size", &vecType::size ).\
        def( "resize", ( void ( vecType::* )( const vecType::size_type ) )& vecType::resize ).\
        def( "clear", &vecType::clear ); \
} )

#define MR_ADD_PYTHON_EXPECTED( moduleName, name, type, errorType )\
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name, [] (pybind11::module_& m)\
{\
    using expectedType = tl::expected<type,errorType>;\
    pybind11::class_<expectedType>(m, #name ).\
        def( "has_value", &expectedType::has_value ).\
        def( "value", ( type& ( expectedType::* )( )& )& expectedType::value, pybind11::return_value_policy::reference ).\
        def( "error", ( const errorType& ( expectedType::* )( )const& )& expectedType::error );\
} )

enum StreamType
{
    Stdout,
    Stderr
};

template<StreamType T>
class PythonStreamRedirector
{
public:
    MRMESH_API void write( const std::string& text );
    void flush() {}
    MRMESH_API static int getNumWritten();
};

using StdoutPyRedirector = PythonStreamRedirector<Stdout>;
using StderrPyRedirector = PythonStreamRedirector<Stderr>;

namespace MR
{

class PythonExport
{
public:
    MRMESH_API static PythonExport& instance();

    using PythonRegisterFuncton = std::function<void( pybind11::module_& m )>;

    struct ModuleData
    {
        PyObject* ( *initFncPointer )( void );
        std::vector<PythonRegisterFuncton> functions;
    };

    void addFunc( const std::string& moduleName, PythonRegisterFuncton func )
    {
        auto& mod = moduleData_[moduleName];
        mod.functions.push_back( func );
    }
    void setInitFuncPtr( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) )
    {
        auto& mod = moduleData_[moduleName];
        mod.initFncPointer = initFncPointer;
    }
    const std::vector<PythonRegisterFuncton> functions( const std::string& moduleName ) const
    {
        auto it = moduleData_.find( moduleName );
        if ( it == moduleData_.end() )
            return {};
        return it->second.functions;
    }

    const std::unordered_map<std::string, ModuleData>& modules() const
    {
        return moduleData_;
    }
private:
    PythonExport() = default;
    ~PythonExport() = default;

     std::unordered_map<std::string, ModuleData> moduleData_;
};

struct PythonFunctionAdder
{
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, std::function<void( pybind11::module_& m )> func );
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) );
};

}
#endif
