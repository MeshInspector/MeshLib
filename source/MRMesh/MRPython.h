#pragma once

#include "MRMeshFwd.h"
#if !defined( __EMSCRIPTEN__) && !defined( MRMESH_NO_PYTHON )
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "MRExpected.h"
#include <functional>
#include <filesystem>
#include <unordered_map>

#define MR_INIT_PYTHON_MODULE( moduleName ) MR_INIT_PYTHON_MODULE_PRECALL( moduleName, [](){} )

#define MR_INIT_PYTHON_MODULE_PRECALL( moduleName, precall )\
PYBIND11_MODULE( moduleName, m )\
{\
    precall();\
    auto& adders = MR::PythonExport::instance().functions( #moduleName );\
    for ( auto& fs : adders )\
        for ( auto& f : fs )\
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

#define MR_ADD_PYTHON_CUSTOM_CLASS_DECL_0( moduleName, name, type ) \
static std::optional<pybind11::class_<type>> name##_class_;

#define MR_ADD_PYTHON_CUSTOM_CLASS_DECL_1( moduleName, name, type, ... ) \
static std::optional<pybind11::class_<type, __VA_ARGS__>> name##_class_;

#define MR_ADD_PYTHON_CUSTOM_CLASS_INST_0( moduleName, name ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name##_inst_, [] ( pybind11::module_& module ) \
{                                                             \
    name##_class_.emplace( module, #name );                   \
}, MR::PythonExport::Priority::Declaration )

#define MR_ADD_PYTHON_CUSTOM_CLASS_INST_1( moduleName, name, ... ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name##_inst_, [] ( pybind11::module_& module ) \
{                                                                  \
    name##_class_ = __VA_ARGS__ ( module );                        \
}, MR::PythonExport::Priority::Declaration )

#define MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, type ) \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL_0( moduleName, name, type )       \
MR_ADD_PYTHON_CUSTOM_CLASS_INST_0( moduleName, name )

#define MR_ADD_PYTHON_CUSTOM_CLASS_IMPL( moduleName, name, ... ) \
MR_ADD_PYTHON_CUSTOM_DEF( moduleName, name##_impl_, [] ( pybind11::module_& ) \
{                                                                \
    __VA_ARGS__ ( *name##_class_ );                              \
}, MR::PythonExport::Priority::Implementation )

// !!! It's important to add vec after adding type
// otherwise embedded python will not be able to re-import module (due to some issues with vector types in pybind11)
#define MR_ADD_PYTHON_VEC( moduleName, name, type) \
PYBIND11_MAKE_OPAQUE( std::vector<type> )          \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL_1( moduleName, name, std::vector<type>, std::unique_ptr<std::vector<type>> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST_1( moduleName, name, [] ( pybind11::module_& module ) { return pybind11::bind_vector<std::vector<type>>( module, #name ); } ) \
MR_ADD_PYTHON_CUSTOM_CLASS_IMPL( moduleName, name, [] ( auto& cls )                                          \
{\
    using vecType = std::vector<type>;\
    cls.\
        def( pybind11::init<>() ).\
        def( pybind11::init<size_t>(), pybind11::arg( "size" ) ).\
        def( "empty", &vecType::empty ).\
        def( "size", &vecType::size ).\
        def( "resize", ( void ( vecType::* )( const vecType::size_type ) )& vecType::resize ).\
        def( "clear", &vecType::clear ); \
} )

#define MR_ADD_PYTHON_MAP( moduleName, name, mapType ) \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL_1( moduleName, name, mapType, std::unique_ptr<mapType> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_INST_1( moduleName, name, [] ( pybind11::module_& module ) { return pybind11::bind_map<mapType>( module, #name ); } ) \
MR_ADD_PYTHON_CUSTOM_CLASS_IMPL( moduleName, name, [] ( auto& cls )                      \
{\
    cls.\
        def( pybind11::init<>() ).\
        def( "size", &mapType::size );\
} )

#define MR_ADD_PYTHON_EXPECTED( moduleName, name, type, errorType ) \
MR_ADD_PYTHON_CUSTOM_CLASS_DECL( moduleName, name, MR::Expected<type, errorType> ) \
MR_ADD_PYTHON_CUSTOM_CLASS_IMPL( moduleName, name, [] ( auto& cls ) \
{\
    using expectedType = Expected<type,errorType>;\
    cls.\
        def( "has_value", []() \
        { PyErr_WarnEx(PyExc_DeprecationWarning, ".has_value is deprecated. Please use 'try - except ValueError'", 1); \
            return &expectedType::has_value; \
        }).\
        def( "value", []() \
        { \
            PyErr_WarnEx(PyExc_DeprecationWarning, ".value is deprecated. Please use 'try - except ValueError'", 1); \
            return ( type& ( expectedType::* )( )& )& expectedType::value; \
        }, pybind11::return_value_policy::reference_internal ).\
        def( "error", []() \
        { \
            PyErr_WarnEx(PyExc_DeprecationWarning, ".error is deprecated. Please use 'try - except ValueError'", 1); \
            return ( const errorType& ( expectedType::* )( )const& )& expectedType::error; \
        } );\
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

    enum class Priority
    {
        Declaration,
        Implementation,
        Count,
    };

    struct ModuleData
    {
        PyObject* ( *initFncPointer )( void );
        std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )> functions;
    };

    void addFunc( const std::string& moduleName, PythonRegisterFuncton func, Priority priority )
    {
        auto& mod = moduleData_[moduleName];
        mod.functions[size_t( priority )].push_back( func );
    }
    void setInitFuncPtr( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) )
    {
        auto& mod = moduleData_[moduleName];
        mod.initFncPointer = initFncPointer;
    }
    const std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )>& functions( const std::string& moduleName ) const
    {
        auto it = moduleData_.find( moduleName );
        const static std::array<std::vector<PythonRegisterFuncton>, size_t( Priority::Count )> empty;
        if ( it == moduleData_.end() )
            return empty;
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
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, std::function<void( pybind11::module_& m )> func, PythonExport::Priority priority = PythonExport::Priority::Implementation );
    MRMESH_API PythonFunctionAdder( const std::string& moduleName, PyObject* ( *initFncPointer )( void ) );
};

// overload `toString` functoion to throw exception from custom `Expected::error` type
template<typename E>
void throwExceptionFromExpected(const E& err)
{
    if constexpr (std::is_nothrow_convertible<E, std::string>::value)
         throw std::runtime_error(err);
    else
        throw std::runtime_error(toString(err));
}

template<typename R, typename E, typename... Args>
auto decorateExpected( std::function<Expected<R, E>( Args... )>&& f ) -> std::function<R( Args... )>
{
    return[fLocal = std::move( f )]( Args&&... args ) mutable -> R
    {
        auto res = fLocal(std::forward<Args>( args )...);
        if (!res.has_value())
            throwExceptionFromExpected(res.error());

        if constexpr (std::is_void<R>::value)
            return;
        else
            return res.value();
    };
}

template<typename F>
auto decorateExpected( F&& f )
{
    return decorateExpected( std::function( std::forward<F>( f ) ) );
}

template<typename R, typename T, typename... Args>
auto decorateExpected( R( T::* memFunction )( Args... ) )
{
    return decorateExpected( std::function<R( T*, Args... )>( std::mem_fn( memFunction ) ) );
}

}
#endif
