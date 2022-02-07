#pragma once

#ifndef __EMSCRIPTEN__
#include "MRPython.h"

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

namespace MR
{

struct PythonFunctionAdder
{
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, std::function<void( pybind11::module_& m )> func );
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) );
};

class MRMESH_CLASS EmbeddedPython
{
public:
    static MRMESH_API void init();

    static MRMESH_API bool isAvailable();

    static MRMESH_API bool isInitialized();

    static MRMESH_API void finalize();

    static MRMESH_API bool setupArgv( int argc, char** argv );

    static MRMESH_API bool runString( const std::string& pythonString );

    static MRMESH_API bool runScript( const std::filesystem::path& path );

    static MRMESH_API bool isPythonScript( const std::filesystem::path& path );
private:
    EmbeddedPython();
    ~EmbeddedPython() = default;

    static EmbeddedPython& instance_();
    bool available_{ false };
};

} //namespace MR
#endif
