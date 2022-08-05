#pragma once

#ifndef __EMSCRIPTEN__
#include "exports.h"
#include <string>
#include <filesystem>

namespace MR
{

class MRMESHPY_CLASS EmbeddedPython
{
public:
    static MRMESHPY_API void init();

    static MRMESHPY_API bool isAvailable();

    static MRMESHPY_API bool isInitialized();

    static MRMESHPY_API void finalize();

    static MRMESHPY_API bool setupArgv( int argc, char** argv );

    static MRMESHPY_API bool runString( const std::string& pythonString );

    static MRMESHPY_API bool runScript( const std::filesystem::path& path );

    static MRMESHPY_API bool isPythonScript( const std::filesystem::path& path );
private:
    EmbeddedPython();
    ~EmbeddedPython() = default;

    static EmbeddedPython& instance_();
    bool available_{ false };
};

} //namespace MR

#endif
