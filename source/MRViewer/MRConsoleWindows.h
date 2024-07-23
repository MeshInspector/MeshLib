#pragma once
#include "exports.h"
#include <vector>
#include <string>

namespace MR
{

#ifdef _WIN32
// convert winapi representation of argv to simple strings vector
MRVIEWER_API std::vector<std::string> ConvertArgv();

// starts console in ctor if needed
// free it in destructor
struct MRVIEWER_CLASS ConsoleRunner
{
    MRVIEWER_API ConsoleRunner( bool runConsole );
    MRVIEWER_API ~ConsoleRunner();
private:
    bool consoleStarted_{ false };
};
#endif //_WIN32

}