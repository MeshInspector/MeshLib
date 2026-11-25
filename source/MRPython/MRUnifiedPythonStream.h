#pragma once
#include "exports.h"
#include <string>
#include <string_view>

namespace MR
{
// python output duplicated to this stream
// needed to be able to separate python output from all other logs;
// thread-safe: Python threads may append while other threads read or clear it
class MRPYTHON_CLASS UnifiedPythonStream
{
public:
    MRPYTHON_API static void append( std::string_view text );
    // returns a copy of all text appended since the last clear()
    MRPYTHON_API static std::string str();
    MRPYTHON_API static void clear();
};
}
