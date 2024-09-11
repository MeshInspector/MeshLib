#pragma once

#include "exports.h"

#include <string>
#include <filesystem>

namespace MR
{

class MRPYTHON_CLASS EmbeddedPython
{
public:
    static MRPYTHON_API void init();

    static MRPYTHON_API bool isAvailable();

    static MRPYTHON_API bool isInitialized();

    static MRPYTHON_API void finalize();

    static MRPYTHON_API bool setupArgv( int argc, char** argv );

    static MRPYTHON_API bool runString( const std::string& pythonString );

    static MRPYTHON_API bool runScript( const std::filesystem::path& path );

    static MRPYTHON_API bool isPythonScript( const std::filesystem::path& path );
private:
    EmbeddedPython();
    ~EmbeddedPython() = default;

    static EmbeddedPython& instance_();
    bool available_{ false };
};

} //namespace MR
