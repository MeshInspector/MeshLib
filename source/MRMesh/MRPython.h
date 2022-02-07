#pragma once

#ifndef __EMSCRIPTEN__
#include "MRMeshFwd.h"
#include <functional>
#include <filesystem>
#include <unordered_map>

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

}
#endif
