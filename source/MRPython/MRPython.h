#pragma once

#include "exports.h"

#include <string>

// The targets of `sys.stdout` and `sys.stderr` in the embedded interpreter (`MREmbeddedPython`).
enum StreamType
{
    Stdout,
    Stderr
};

template<StreamType T>
class PythonStreamRedirector
{
public:
    MRPYTHON_API void write( const std::string& text );
    MRPYTHON_API void flush();
    MRPYTHON_API static int getNumWritten();
};

using StdoutPyRedirector = PythonStreamRedirector<Stdout>;
using StderrPyRedirector = PythonStreamRedirector<Stderr>;
