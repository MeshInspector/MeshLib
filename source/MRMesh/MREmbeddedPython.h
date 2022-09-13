#pragma once

#ifndef MRMESH_NO_PYTHON
#include "MRMeshFwd.h"
#include <string>
#include <filesystem>

namespace MR
{

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
